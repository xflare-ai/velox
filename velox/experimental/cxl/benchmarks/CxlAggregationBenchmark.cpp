/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/time/Timer.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/experimental/cxl/CxlMemoryResource.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

DEFINE_string(
    config,
    "dram",
    "Placement configuration: dram | interleave | cxl | cxl_migrate.");
DEFINE_int64(zipf_groups, 1'000'000, "Number of distinct grouping keys.");
DEFINE_double(
    zipf_skew,
    1.0,
    "Zipf exponent: rank r is drawn with probability proportional to "
    "1/r^zipf_skew. 0 is uniform; higher is more skewed.");
DEFINE_double(
    scale_factor,
    1.0,
    "Input size: scale_factor = 1 is ~1GB of (key, value) data.");
DEFINE_int64(
    dram_limit_mb,
    48,
    "Query DRAM pool capacity in MB for the 'dram' and 'cxl' configs. Set "
    "below the group-table size to force spill / relocation.");
DEFINE_int32(
    cxl_numa_node,
    -1,
    "NUMA node id of the CXL device (required for --config=cxl).");
DEFINE_int64(
    cxl_capacity_mb,
    0,
    "CXL pool capacity in MB (required for --config=cxl; size it to the CXL "
    "device). The allocator pre-reserves this, so it must be bounded.");
DEFINE_int32(num_trials, 5, "Number of measured trials.");
DEFINE_int32(warmup, 1, "Number of warmup trials to discard.");
DEFINE_int32(
    num_drivers,
    1,
    "Local parallelism for the aggregation. 1 runs the serial single-stage "
    "plan; >1 splits the input across that many source pipelines, "
    "repartitions by key, and aggregates on that many drivers.");
DEFINE_string(
    spill_dir,
    "/tmp/cxl_bench_spill",
    "Spill directory for 'dram' and 'cxl'.");
DEFINE_bool(
    migrate_buckets,
    false,
    "For --config=cxl_migrate, also migrate the hash bucket array to CXL, not "
    "just the row payload.");
DEFINE_int64(
    allocator_capacity_gb,
    64,
    "Total MmapAllocator capacity in GB, shared by the input and query pools. "
    "Raise it for large --num_drivers sweeps where the scaled input exceeds the "
    "default.");

using namespace facebook::velox;
using exec::test::PlanBuilder;

namespace {

constexpr int32_t kBatchSize = 4'096;

struct TrialMetrics {
  uint64_t elapsedMs{0};
  uint64_t aggCpuNanos{0};
  uint64_t aggWallNanos{0};
  uint64_t aggBlockedNanos{0};
  uint64_t aggPeakBytes{0};
  uint64_t numGroups{0};
  uint64_t spilledBytes{0};
  uint64_t spilledRows{0};
  uint64_t spillWriteNanos{0};
  uint64_t spillReadNanos{0};
  uint64_t relocatedBytes{0};
  uint64_t migrateWallNanos{0};
  int64_t resultRows{0};
  uint64_t checksum{0};
};

bool isCxlCopyConfig() {
  return FLAGS_config == "cxl";
}

bool isCxlMigrateConfig() {
  return FLAGS_config == "cxl_migrate";
}

bool usesCxlTier() {
  return isCxlCopyConfig() || isCxlMigrateConfig();
}

int64_t dramCapacityBytes() {
  if (FLAGS_config == "dram" || usesCxlTier()) {
    return FLAGS_dram_limit_mb << 20;
  }
  return memory::kMaxMemory;
}

std::vector<std::vector<RowVectorPtr>> splitInputRoundRobin(
    const std::vector<RowVectorPtr>& input,
    int32_t numChunks) {
  std::vector<std::vector<RowVectorPtr>> chunks(numChunks);
  for (auto i = 0; i < input.size(); ++i) {
    chunks[i % numChunks].push_back(input[i]);
  }
  return chunks;
}

core::PlanNodePtr buildSerialPlan(const std::vector<RowVectorPtr>& input) {
  return PlanBuilder()
      .values(input)
      .singleAggregation({"k"}, {"sum(v) AS s"})
      .planNode();
}

core::PlanNodePtr buildKeyPartitionedPlan(
    const std::vector<RowVectorPtr>& input) {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto chunks = splitInputRoundRobin(input, FLAGS_num_drivers);
  std::vector<core::PlanNodePtr> sources;
  sources.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    sources.push_back(
        PlanBuilder(planNodeIdGenerator).values(chunk).planNode());
  }
  return PlanBuilder(planNodeIdGenerator)
      .localPartition({"k"}, sources)
      .singleAggregation({"k"}, {"sum(v) AS s"})
      .planNode();
}

core::PlanNodePtr buildPlan(const std::vector<RowVectorPtr>& input) {
  return FLAGS_num_drivers <= 1 ? buildSerialPlan(input)
                                : buildKeyPartitionedPlan(input);
}

void accumulateResult(
    const std::vector<RowVectorPtr>& results,
    TrialMetrics& metrics) {
  for (const auto& result : results) {
    if (result == nullptr || result->size() == 0) {
      continue;
    }
    metrics.resultRows += result->size();
    SelectivityVector rows(result->size());
    DecodedVector decoded(*result->childAt(0), rows);
    for (auto i = 0; i < result->size(); ++i) {
      if (!decoded.isNullAt(i)) {
        metrics.checksum += static_cast<uint64_t>(decoded.valueAt<int64_t>(i));
      }
    }
  }
}

uint64_t runtimeSum(
    const std::unordered_map<std::string, RuntimeMetric>& stats,
    std::string_view key) {
  const auto it = stats.find(std::string(key));
  return it == stats.end() ? 0 : static_cast<uint64_t>(it->second.sum);
}

bool isAggregationOperator(const exec::OperatorStats& op) {
  return op.operatorType.find("Aggregation") != std::string::npos;
}

void accumulateAggregationStats(
    const exec::OperatorStats& op,
    TrialMetrics& metrics) {
  metrics.aggCpuNanos += op.addInputTiming.cpuNanos +
      op.getOutputTiming.cpuNanos + op.finishTiming.cpuNanos;
  metrics.aggWallNanos += op.addInputTiming.wallNanos +
      op.getOutputTiming.wallNanos + op.finishTiming.wallNanos;
  metrics.aggBlockedNanos += op.blockedWallNanos;
  metrics.aggPeakBytes = std::max<uint64_t>(
      metrics.aggPeakBytes, op.memoryStats.peakTotalMemoryReservation);
  metrics.numGroups += op.outputPositions;
  metrics.spilledBytes += op.spilledBytes;
  metrics.spilledRows += op.spilledRows;
  metrics.spillWriteNanos += runtimeSum(op.runtimeStats, "spillWriteWallNanos");
  metrics.spillReadNanos += runtimeSum(op.runtimeStats, "spillReadWallNanos");
  metrics.relocatedBytes +=
      runtimeSum(op.runtimeStats, memory::kRelocatedMemoryBytes);
  metrics.migrateWallNanos +=
      runtimeSum(op.runtimeStats, "cxlMigrateWallNanos");
}

void collectOperatorStats(const exec::TaskStats& stats, TrialMetrics& metrics) {
  metrics.elapsedMs = stats.executionEndTimeMs - stats.executionStartTimeMs;
  for (const auto& pipeline : stats.pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      if (isAggregationOperator(op)) {
        accumulateAggregationStats(op, metrics);
      }
    }
  }
}

int64_t targetRowCount() {
  constexpr int64_t kInputBytesPerRow = sizeof(int64_t) * 2;
  return static_cast<int64_t>(FLAGS_scale_factor * (1LL << 30)) /
      kInputBytesPerRow;
}

std::vector<double> buildZipfCdf(int64_t numGroups, double skew) {
  std::vector<double> cdf(numGroups);
  double total = 0;
  for (int64_t rank = 0; rank < numGroups; ++rank) {
    total += 1.0 / std::pow(rank + 1, skew);
    cdf[rank] = total;
  }
  for (auto& value : cdf) {
    value /= total;
  }
  return cdf;
}

int64_t pickRank(
    int64_t globalRow,
    int64_t numGroups,
    const std::vector<double>& cdf,
    std::mt19937_64& rng,
    std::uniform_real_distribution<double>& uniform) {
  if (globalRow < numGroups) {
    return globalRow;
  }
  return std::upper_bound(cdf.begin(), cdf.end(), uniform(rng)) - cdf.begin();
}

int64_t scatterKey(int64_t rank) {
  return static_cast<int64_t>(
      static_cast<uint64_t>(rank + 1) * 0x9E3779B97F4A7C15ULL);
}

RowVectorPtr makeZipfBatch(
    int64_t begin,
    vector_size_t size,
    int64_t numGroups,
    const std::vector<double>& cdf,
    const RowTypePtr& rowType,
    memory::MemoryPool* pool) {
  std::mt19937_64 rng(42 + begin);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  auto keys = BaseVector::create<FlatVector<int64_t>>(BIGINT(), size, pool);
  auto values = BaseVector::create<FlatVector<int64_t>>(BIGINT(), size, pool);
  for (vector_size_t i = 0; i < size; ++i) {
    const int64_t rank = pickRank(begin + i, numGroups, cdf, rng, uniform);
    keys->set(i, scatterKey(rank));
    values->set(i, static_cast<int64_t>(uniform(rng) * 100));
  }
  return std::make_shared<RowVector>(
      pool,
      rowType,
      nullptr,
      size,
      std::vector<VectorPtr>{std::move(keys), std::move(values)});
}

int64_t fillBatchesInParallel(
    std::vector<RowVectorPtr>& batches,
    int64_t numRows,
    int64_t numGroups,
    const std::vector<double>& cdf,
    const RowTypePtr& rowType,
    memory::MemoryPool* pool) {
  const int64_t numBatches = batches.size();
  const int64_t numThreads = std::min<int64_t>(
      numBatches, std::max<unsigned>(1u, std::thread::hardware_concurrency()));
  const int64_t batchesPerThread = (numBatches + numThreads - 1) / numThreads;
  std::vector<int64_t> threadBytes(numThreads, 0);
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for (int64_t t = 0; t < numThreads; ++t) {
    const int64_t firstBatch = t * batchesPerThread;
    const int64_t lastBatch =
        std::min<int64_t>(firstBatch + batchesPerThread, numBatches);
    if (firstBatch >= lastBatch) {
      break;
    }
    threads.emplace_back([&, t, firstBatch, lastBatch]() {
      for (int64_t batch = firstBatch; batch < lastBatch; ++batch) {
        const int64_t begin = batch * kBatchSize;
        const auto size = static_cast<vector_size_t>(
            std::min<int64_t>(kBatchSize, numRows - begin));
        batches[batch] =
            makeZipfBatch(begin, size, numGroups, cdf, rowType, pool);
        threadBytes[t] += batches[batch]->retainedSize();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  int64_t numBytes = 0;
  for (const auto bytes : threadBytes) {
    numBytes += bytes;
  }
  return numBytes;
}

void printGenerationSummary(
    int64_t numRows,
    int64_t numGroups,
    size_t numBatches,
    int64_t numBytes,
    int64_t elapsedMs) {
  std::cout << fmt::format(
                   "generated {} zipf rows ({} groups, skew {}) in {} "
                   "batches, {:.1f} MB, {} ms (excluded from trials)\n",
                   numRows,
                   numGroups,
                   FLAGS_zipf_skew,
                   numBatches,
                   numBytes / static_cast<double>(1 << 20),
                   elapsedMs)
            << std::flush;
}

std::vector<RowVectorPtr> generateZipfInput(
    const std::shared_ptr<memory::MemoryPool>& outputPool) {
  const int64_t numRows = targetRowCount();
  const int64_t numGroups = FLAGS_zipf_groups;
  const auto startMs = getCurrentTimeMs();
  const auto cdf = buildZipfCdf(numGroups, FLAGS_zipf_skew);
  const auto rowType = ROW({"k", "v"}, {BIGINT(), BIGINT()});
  const int64_t numBatches = (numRows + kBatchSize - 1) / kBatchSize;
  std::vector<RowVectorPtr> batches(numBatches);
  const int64_t numBytes = fillBatchesInParallel(
      batches, numRows, numGroups, cdf, rowType, outputPool.get());
  printGenerationSummary(
      numRows, numGroups, batches.size(), numBytes, getCurrentTimeMs() - startMs);
  return batches;
}

std::shared_ptr<core::QueryCtx> makeQueryCtx(
    const std::string& queryId,
    folly::Executor* executor,
    std::shared_ptr<memory::CustomMemoryResource>& cxlResource) {
  auto* manager = memory::memoryManager();
  auto rootPool = manager->addRootPool(
      queryId, dramCapacityBytes(), memory::MemoryReclaimer::create());
  auto builder =
      core::QueryCtx::Builder().executor(executor).pool(rootPool).queryId(
          queryId);
  if (usesCxlTier()) {
    cxlResource = cxl::CxlMemoryResource::create(
        FLAGS_cxl_numa_node, FLAGS_cxl_capacity_mb << 20);
    auto cxlPool = manager->addCustomRootPool(queryId + ".cxl", cxlResource);
    builder.customPool(
        std::string{cxl::CxlMemoryResource::kTag}, std::move(cxlPool));
  }
  auto queryCtx = builder.build();
  if (usesCxlTier()) {
    auto registry =
        memory::CustomMemoryResourceRegistry::createRegistry(nullptr);
    queryCtx->setRegistry<memory::CustomMemoryResourceRegistry::Registry>(
        memory::kCustomMemoryResourceRegistryKey, registry);
    registry->insert(std::string{cxl::CxlMemoryResource::kTag}, cxlResource);
  }
  return queryCtx;
}

exec::CursorParameters makeCursorParameters(
    const core::PlanNodePtr& plan,
    const std::shared_ptr<core::QueryCtx>& queryCtx) {
  exec::CursorParameters params;
  params.planNode = plan;
  params.queryCtx = queryCtx;
  params.maxDrivers = std::max(1, FLAGS_num_drivers);
  params.copyResult = true;
  if (FLAGS_config == "dram" || usesCxlTier()) {
    params.spillDirectory = FLAGS_spill_dir;
    params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
    params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
  }
  if (usesCxlTier()) {
    params.queryConfigs[core::QueryConfig::kRelocationResourceTag] =
        std::string{cxl::CxlMemoryResource::kTag};
  }
  if (isCxlMigrateConfig()) {
    params.queryConfigs[core::QueryConfig::kRelocationMode] = "migrate";
    params.queryConfigs[core::QueryConfig::kRelocationMigrateNumaNode] =
        std::to_string(FLAGS_cxl_numa_node);
    params.queryConfigs[core::QueryConfig::kRelocationMigrateIncludeBuckets] =
        FLAGS_migrate_buckets ? "true" : "false";
  }
  return params;
}

TrialMetrics runTrial(
    const core::PlanNodePtr& plan,
    folly::Executor* executor,
    int32_t trial) {
  const auto queryId = fmt::format("{}-{}", FLAGS_config, trial);
  std::shared_ptr<memory::CustomMemoryResource> cxlResource;
  auto queryCtx = makeQueryCtx(queryId, executor, cxlResource);
  const auto params = makeCursorParameters(plan, queryCtx);
  auto [cursor, results] = exec::test::readCursor(
      params, [](exec::TaskCursor* cursor) { cursor->setNoMoreSplits(); });
  exec::test::waitForTaskCompletion(
      cursor->task().get(), 600'000'000);
  TrialMetrics metrics;
  collectOperatorStats(cursor->task()->taskStats(), metrics);
  accumulateResult(results, metrics);
  return metrics;
}

double medianMs(std::vector<uint64_t> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void printRunConfig(size_t trialCount) {
  std::cout << "\n=== CXL aggregation benchmark ===\n";
  std::cout << fmt::format(
                   "config={} scale_factor={} zipf_groups={} zipf_skew={} "
                   "dram_limit_mb={} trials={}",
                   FLAGS_config,
                   FLAGS_scale_factor,
                   FLAGS_zipf_groups,
                   FLAGS_zipf_skew,
                   FLAGS_dram_limit_mb,
                   trialCount)
            << "\n";
  if (usesCxlTier()) {
    std::cout << fmt::format(
                     "cxl_numa_node={} cxl_capacity_mb={}",
                     FLAGS_cxl_numa_node,
                     FLAGS_cxl_capacity_mb)
              << "\n";
  }
}

void printAggregation(const TrialMetrics& last) {
  std::cout << fmt::format(
      "aggregation: cpu={:.1f} ms wall={:.1f} ms blocked={:.1f} "
      "ms peak={:.1f} MB\n",
      last.aggCpuNanos / 1e6,
      last.aggWallNanos / 1e6,
      last.aggBlockedNanos / 1e6,
      last.aggPeakBytes / static_cast<double>(1 << 20));
}

void printSpill(const TrialMetrics& last) {
  std::cout << fmt::format(
      "spill: bytes={:.1f} MB rows={} write={:.1f} ms read={:.1f} ms\n",
      last.spilledBytes / static_cast<double>(1 << 20),
      last.spilledRows,
      last.spillWriteNanos / 1e6,
      last.spillReadNanos / 1e6);
}

void printRelocation(const TrialMetrics& last) {
  std::cout << fmt::format(
      "cxl relocated: {:.1f} MB\n",
      last.relocatedBytes / static_cast<double>(1 << 20));
  if (isCxlMigrateConfig()) {
    std::cout << fmt::format(
        "cxl migrate wall: {:.1f} ms\n", last.migrateWallNanos / 1e6);
  }
  if (last.relocatedBytes == 0) {
    std::cout << "WARNING: no relocation fired; the DRAM cap did not trigger "
                 "the arbitrator. Lower --dram_limit_mb or verify the CXL "
                 "pool is set.\n";
  }
}

void printResult(const TrialMetrics& last) {
  std::cout << fmt::format(
                   "result: groups built={} output rows={} checksum={}\n",
                   last.numGroups,
                   last.resultRows,
                   last.checksum)
            << std::flush;
}

void report(const std::vector<TrialMetrics>& trials) {
  std::vector<uint64_t> elapsed;
  for (const auto& trial : trials) {
    elapsed.push_back(trial.elapsedMs);
  }
  const auto& last = trials.back();
  printRunConfig(trials.size());
  std::cout << fmt::format("median elapsed: {:.1f} ms\n", medianMs(elapsed));
  printAggregation(last);
  if (FLAGS_config == "dram" || usesCxlTier()) {
    printSpill(last);
  }
  if (usesCxlTier()) {
    printRelocation(last);
  }
  printResult(last);
}

int validateFlags() {
  if (FLAGS_zipf_groups <= 0) {
    LOG(ERROR) << "--zipf_groups must be > 0.";
    return 1;
  }
  if (usesCxlTier() && FLAGS_cxl_numa_node < 0) {
    LOG(ERROR) << "--config=" << FLAGS_config
               << " requires --cxl_numa_node to be set to the CXL device's "
                  "NUMA node id.";
    return 1;
  }
  if (usesCxlTier() && FLAGS_cxl_capacity_mb <= 0) {
    LOG(ERROR) << "--config=" << FLAGS_config
               << " requires --cxl_capacity_mb > 0; the CXL allocator "
                  "pre-reserves this capacity.";
    return 1;
  }
  return 0;
}

void initializeMemoryManager() {
  memory::SharedArbitrator::registerFactory();
  memory::MemoryManager::Options options;
  options.allocatorCapacity = FLAGS_allocator_capacity_gb << 30;
  options.arbitratorKind = "SHARED";
  options.useMmapAllocator = true;
  memory::MemoryManager::testingSetInstance(options);
}

void registerRuntimeComponents() {
  filesystems::registerLocalFileSystem();
  if (!isRegisteredVectorSerde()) {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde("Presto")) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
}

int runBenchmark(folly::Executor* executor) {
  auto inputPool = memory::memoryManager()->addLeafPool("cxl-bench-input");
  std::vector<RowVectorPtr> input;
  core::PlanNodePtr plan;
  try {
    input = generateZipfInput(inputPool);
    plan = buildPlan(input);
    for (auto i = 0; i < FLAGS_warmup; ++i) {
      runTrial(plan, executor, -1 - i);
    }
    std::vector<TrialMetrics> trials;
    for (auto i = 0; i < FLAGS_num_trials; ++i) {
      trials.push_back(runTrial(plan, executor, i));
    }
    report(trials);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Benchmark config '" << FLAGS_config
               << "' failed: " << e.what();
    exec::test::waitForAllTasksToBeDeleted(30'000'000);
    return 1;
  }
  exec::test::waitForAllTasksToBeDeleted(30'000'000);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  if (const int rc = validateFlags(); rc != 0) {
    return rc;
  }
  initializeMemoryManager();
  registerRuntimeComponents();
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(
      std::thread::hardware_concurrency());
  return runBenchmark(executor.get());
}
