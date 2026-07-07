# CXL hash-aggregation benchmark

Measures one grouping aggregation

```sql
SELECT k, sum(v) FROM <synthesized> GROUP BY k
```

over a synthesized Zipf key stream, across three memory-placement strategies, to
see whether building the table in DRAM and relocating it to CXL under pressure
(the `HashAggregation` relocate path, selected by a `cxl` tier pool on the query)
beats the alternatives. Keys are drawn Zipf(`--zipf_skew`)
over `--zipf_groups` ranks and scattered across the key space, so a few hot
groups take most updates in random arrival order — the hot/cold split that
DRAM-to-CXL tiering targets. A seed pass emits each of the `--zipf_groups` ranks
exactly once before sampling, so the whole key space is always populated and the
group-table footprint is held constant across skew — skew then varies only the
access concentration, not how many groups exist (this requires rows ≥ groups).
`--scale_factor` sizes the input: `1` is ~1 GB of `(k, v)` rows (16 B/row). The
plan is driven Velox-local (`Values` → aggregation), single-driver by default;
`--num_drivers=N` repartitions the input by key across `N` drivers for a parallel
run. Every config is apples-to-apples in one process.

## Configurations

| Flag | Role | Memory |
| --- | --- | --- |
| `--config=dram` | A — no-CXL competitor | DRAM pool capped; on-disk spill |
| `--config=interleave` | B — OS striping | DRAM+CXL interleaved by `numactl` |
| `--config=cxl` | C — copy+repoint | DRAM capped; byte-copy overflow to CXL |
| `--config=cxl_migrate` | D — move_pages | DRAM capped; migrate overflow pages to CXL |

C and D are the A/B for the relocation mechanism: same plan, cap, and input,
differing only in how overflow reaches CXL. C byte-copies the payload into the
CXL pool and repoints the hash index; D (`relocation_mode=migrate`) migrates the
backing pages in place with `move_pages`, leaving virtual addresses and the
index unchanged. D reports `cxl migrate wall` (the move_pages time); compare it
and total elapsed against C to see whether copy+repoint's edge on the
microbenchmark holds on the real `HashTable` and at multi-driver (`--num_drivers`,
where migration's TLB-shootdown / kernel-lock cost scales with cores). D requires
`--cxl_numa_node`; add `--migrate_buckets` to also migrate the bucket array.

Because `move_pages` relocates physical pages but leaves the allocations owned by
the DRAM pool, D relieves the DRAM cap during reclaim with `reportExternalFree`
and restores the accounting at operator close under a lifted cap — VA-stable
migration does not fit Velox's per-pool byte-cap arbitration as cleanly as the
copy path, which is itself a result worth noting.

A and C share the same DRAM cap (`--dram_limit_mb`), differing only in overflow
target (disk vs CXL); the driver script sweeps the cap (`DRAM_MB_LIST`) so the
cap's effect on each is visible. For the no-pressure DRAM speed ceiling, run
`--config=dram` with a `--dram_limit_mb` above the whole group table: the table
fits, nothing spills.

B's DRAM share is controlled differently: a Velox byte cap would not change where
the kernel places pages, so the script sweeps the **placement ratio** instead,
via weighted interleave (`WEIGHTS_LIST`, e.g. `"4:1 2:1 1:1 1:2"` dram:cxl). This
needs Linux 6.9+ (`/sys/kernel/mm/mempolicy/weighted_interleave`) and a numactl
with `--weighted-interleave`; weights are global sysfs settings and writing them
requires root. Without support, B runs only the classic 1:1 stripe.

The two sweeps line up: B at ratio `r` spends `r × footprint` bytes of DRAM
spread uniformly over every page, while C at `--dram_limit_mb ≈ r × footprint`
spends the same DRAM budget structurally (bucket array + not-yet-relocated rows).
Comparing the matched pairs isolates what semantic placement buys over blind
striping at an equal DRAM budget.

## 1. Bring the CXL device online as a NUMA node

```bash
sudo scripts/cxl_numa_setup.sh dax0.0
```

This reconfigures the CXL expander to `system-ram` and prints the resulting
CPU-less NUMA node id. The same node is used for `--membind` (config C's pool
binds itself there) and for `--interleave=0,<node>` (config B) — `membind` and
`interleave` are different `numactl` policies over the same node, not different
setups.

## 2. Build

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DVELOX_ENABLE_CXL=ON \
  -DVELOX_ENABLE_BENCHMARKS=ON <other flags> .
make velox_cxl_aggregation_benchmark
```

Linux only. Requires the libnuma **development** package — `libnuma-dev`
(Debian/Ubuntu) or `numactl-devel` (RHEL/CentOS) — for the `numa.h` header and
the `libnuma.so` symlink that CMake's `find_library(numa)` resolves; the runtime
`libnuma1` alone is not enough, and configure aborts with a `libnuma not found`
error. **Benchmark only release builds** — debug numbers are 5-20x inflated and
meaningless.

Alternatively, build via the Makefile target, which wires the flags for you:

```bash
make benchmarks-build EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_CXL=ON"
```

`VELOX_ENABLE_BENCHMARKS=ON` auto-enables `VELOX_BUILD_TEST_UTILS`, so
`velox_exec_test_lib` (the `PlanBuilder` harness the benchmark links) is built
even though this target sets `VELOX_BUILD_TESTING=OFF`.

## 3. Run

```bash
BENCH=_build/release/velox/experimental/cxl/benchmarks/velox_cxl_aggregation_benchmark \
CXL_NODE=2 DRAM_NODE=0 SF=1 \
DRAM_MB_LIST="32 40 48 56" WEIGHTS_LIST="4:1 2:1 1:1 1:2" CXL_MB=4096 \
EXTRA="--zipf_groups=1000000 --zipf_skew=1.0" \
  ./run_cxl_benchmark.sh
```

This sweeps the interleave ratio over `WEIGHTS_LIST` for config B, then the DRAM
cap over `DRAM_MB_LIST` for the `dram`, `cxl`, and `cxl_migrate` configs (A, C,
D). Restrict the set with `CONFIGS`; e.g. `CONFIGS="C D"` runs just the
copy-vs-migrate A/B and skips the slow spill and interleave legs.

To A/B copy vs move_pages across driver counts (the multi-driver comparison the
maintainer asked for), sweep `--num_drivers` via `EXTRA`:

```bash
for nd in 1 4 8 16; do
  BENCH=... CXL_NODE=2 DRAM_NODE=0 SF=1 CONFIGS="C D" \
  DRAM_MB_LIST="48" CXL_MB=4096 \
  EXTRA="--zipf_groups=1000000 --zipf_skew=1.0 --num_drivers=${nd}" \
    ./run_cxl_benchmark.sh
done
```

Or run a single config directly. Copy+repoint (C):

```bash
numactl --cpunodebind=0 --membind=0 \
  ./velox_cxl_aggregation_benchmark --config=cxl --cxl_numa_node=2 \
  --cxl_capacity_mb=4096 --dram_limit_mb=48
```

move_pages migration (D) — same flags, plus optional `--migrate_buckets` to also
migrate the bucket array:

```bash
numactl --cpunodebind=0 --membind=0 \
  ./velox_cxl_aggregation_benchmark --config=cxl_migrate --cxl_numa_node=2 \
  --cxl_capacity_mb=4096 --dram_limit_mb=48 --num_drivers=8
```

Each run reports median elapsed time plus an aggregation-operator breakdown (CPU,
wall, blocked, peak), spill bytes/time (config A), CXL relocated bytes (C and D),
the move_pages wall time (`cxl migrate wall`, D only), and a result row count +
checksum. Compare C's total elapsed and relocated bytes against D's total elapsed
and migrate wall to see whether copy+repoint's microbenchmark edge holds on the
real `HashTable` and as `--num_drivers` grows.

## Reading the results

The raw log interleaves results with arbitrator stack dumps (see the
`MEM_CAP_EXCEEDED` note below). For a one-line-per-leg table, pipe it through the
summarizer:

```bash
./run_cxl_benchmark.sh ... | scripts/summarize.sh
# or on a saved log:
scripts/summarize.sh results/zipf-sf1.log
```

- **Correctness**: the `checksum` and output `rows` must match across all
  configs.
- **Relocation fired (config C)**: `cxl relocations` must be `> 0`. If it is `0`,
  the DRAM cap did not trigger the arbitrator — lower `--dram_limit_mb`.
- **`MEM_CAP_EXCEEDED` stack traces are expected, not a crash (config C)**: at
  the tightest caps the log fills with full `SharedArbitrator::growCapacity`
  dumps ending in `ErrorCode: MEM_CAP_EXCEEDED`. That failed DRAM grow is
  precisely what triggers relocation to CXL — the run continues and finishes with
  a correct checksum. Treat them as noise unless the process actually exits
  non-zero or `LEG FAILED` is printed by the driver.

## Tuning: the DRAM cap has a floor

Relocation moves row payload to CXL, but the bucket array (`table_`) stays pinned
in DRAM by design (both C and D; D migrates it too only with `--migrate_buckets`).
So `--dram_limit_mb` must sit **above** the bucket-array floor
(the index has to fit) yet **below** the whole group table (or no overflow is
forced). Both scale with `--zipf_groups`. The figures below are starting points; read the
measured `peak` off a `dram` leg with a large `--dram_limit_mb` (the uncapped
ceiling) and bracket the sweep just under it.

| `--zipf_groups` | Bucket-array floor | Whole table | `DRAM_MB_LIST` | `CXL_MB` |
|---:|---:|---:|---|---:|
| 1M (default) | ~16 MB | ~50 MB | `32 40 48 56` | `4096` |
| 4M | ~64 MB | ~200 MB | `100 140 180 220` | `4096` |
| 16M | ~256 MB | ~800 MB | `400 550 700 850` | `8192` |

Pick `--scale_factor` so rows ≫ groups (SF1 is ~67M rows over the default 1M
groups, ~67 hits per group on average and far more on the hot head). The seed
pass requires rows ≥ groups: it emits each group once up front, so the table is
fully populated and its footprint is constant across skew; the remaining rows
sharpen the hot/cold split that distinguishes C from B. With fewer rows than
groups only the first `rows` ranks would be emitted, shrinking the table.
`CXL_MB` must hold the relocated payload (whole table minus the DRAM cap), so
size it to the CXL device.

A cap below the floor is a legitimate `LEG FAILED` for configs C and D (the
bucket array alone will not fit), not a bug.

The benchmark must use the **real** NUMA-bound CXL pool (it fails if
`--cxl_numa_node`/`--cxl_capacity_mb` are unset for `--config=cxl` or
`--config=cxl_migrate`); never wire it to the unit tests' `MallocAllocator`
resource, which is DRAM-speed and would make C's and D's numbers meaningless.
`--cxl_capacity_mb` is pre-reserved by the allocator, so size it to the device.
For D, `--cxl_numa_node` is also the `move_pages` migration target, so it must be
the CXL device's node.
