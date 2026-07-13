# CXL aggregation benchmark — copy+remap vs page migration (multi-driver)

A/B comparison of the two CXL relocation mechanisms for `HashAggregation`, run
at query-level multi-driver parallelism:

- **C — `--config=cxl`**: copy+repoint. Byte-copies the overflow payload into
  the CXL pool and repoints the hash index.
- **D — `--config=cxl_migrate`**: `move_pages`. Migrates the backing pages in
  place, leaving virtual addresses and the index unchanged.

## Setup

Both configs run the same plan, cap, and input, differing only in how overflow
reaches CXL.

- Build: release (`-DVELOX_ENABLE_CXL=ON`).
- NUMA binding: compute + DRAM bound to node 2; the CXL pool bound to node 5
  (a CPU-less `system-ram` NUMA node backed by a real CXL device, `dax12.0`).
- Query: `SELECT k, sum(v) FROM <synthesized> GROUP BY k` over a Zipf(1.0) key
  stream.
- 5 trials per leg; median elapsed reported.

### Multi-driver scaling

The DRAM cap is a single query-level pool shared across all drivers, so a flat
cap starves the drivers as their count grows. To keep the per-driver workload
constant, everything is scaled by the driver count `N`:

| Parameter | Value | Rationale |
| --- | --- | --- |
| `--dram_limit_mb` | `48 × N` | Each driver gets a ~48 MB share of the shared cap. |
| `--zipf_groups` | `1e6 × N` | Each driver owns ~1M groups, so its partition overflows its cap share and relocates. |
| `--scale_factor` | `N` | ~67M rows per driver (rows ≥ groups; ~67 rows/group). |

Each driver therefore reproduces the single-driver workload.

## Results

| drivers | groups | cap | C (copy+remap) elapsed / relocated | D (migration) elapsed / migrate wall |
| ---: | ---: | ---: | --- | --- |
| 1 | 1 M | 48 MB | 2 414 ms / 28 MB | 2 333 ms / 9.8 ms |
| 8 | 8 M | 384 MB | 7 354 ms / 140 MB | 7 628 ms / 45 ms |
| 16 | 16 M | 768 MB | 15 759 ms / 196 MB | 15 842 ms / 69 ms |
| 64 | 64 M | 3072 MB | 20 414 ms / 784 MB | 20 004 ms / 267 ms |
| 128 | 128 M | 6144 MB | 43 693 ms / 3080 MB | 44 282 ms / 446 ms |

All legs completed with matching output row counts and checksums between C and D
at each driver count.

> **Reading the two metrics.** C reports relocated *bytes* — the copy path shows
> up directly in pool accounting. D reports migrate *wall time*. Because
> `move_pages` is VA-stable, it does not change pool accounting, so D's
> relocated-bytes counter reads 0 even though the pages do move; the migrate wall
> is its meaningful cost.

## Conclusion

Page migration (D) does not regress against copy+remap (C): the two are within a
few percent of each other at every driver count, up to 128 drivers.

This differs from the microbenchmark, where migration was slower, for two
reasons:

1. Velox uses transparent huge pages, so migration costs only ~1.5× copy per
   relocation — bounded, not catastrophic.
2. Relocation fires once per memory-limit hit, not per row. Even at 128 drivers
   the total migration time (`migrate wall`) is ~1% of query elapsed (446 ms of
   44 s), so it is in the noise at query scale.

## Note

Reaching a crash-free multi-driver migration run required a teardown fix:
`HashAggregation::restoreMigrationAccounting()` entered memory arbitration during
`close()` and aborted the process when a peer driver's `MEM_CAP_EXCEEDED` had
already terminated the task (`Terminate detected when entering suspension`). The
re-charge now runs under a `ScopedMemoryArbitrationContext`, admitting the
reservation as controlled overuse instead of entering arbitration.
