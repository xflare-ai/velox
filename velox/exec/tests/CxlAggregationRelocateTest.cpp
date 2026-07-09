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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/AllocationPool.h"
#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Spill.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/experimental/cxl/CxlMemoryResource.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::cxl {
namespace {

// A memory tier resource tagged "cxl" but backed by MallocAllocator, so the
// relocation seam runs on hardware without real CXL devices. The libnuma path
// is covered by CxlMemoryResourceTest (skipped when NUMA is unavailable); here
// we only need a custom pool the stock aggregation can resolve by tag.
std::shared_ptr<memory::CustomMemoryResource> makeEmulatedTierResource() {
  memory::MemoryAllocator::Options options;
  options.capacity = 1L << 30;
  return std::make_shared<memory::CustomMemoryResource>(
      std::string{CxlMemoryResource::kTag},
      std::make_shared<memory::MallocAllocator>(options),
      memory::MemoryArbitrator::create({}),
      [] { return memory::MemoryReclaimer::create(0); });
}

class CxlAggregationRelocateTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
  }

  // Builds a QueryCtx carrying a per-query "cxl" custom root pool plus an
  // isolated registry, so the Task can resolve the resource when mirroring the
  // custom pool tree and the operator can reach its tier leaf via
  // customPool("cxl"). No disk spill is configured: the tier is the only
  // reclaim destination.
  std::shared_ptr<core::QueryCtx> makeQueryCtxWithTier(
      const std::string& queryId) {
    auto* manager = memory::memoryManager();
    auto resource = makeEmulatedTierResource();
    auto pool =
        manager->addCustomRootPool(fmt::format("{}.cxl", queryId), resource);
    auto queryCtx =
        core::QueryCtx::Builder()
            .executor(driverExecutor_.get())
            .customPool(std::string{CxlMemoryResource::kTag}, std::move(pool))
            .queryId(queryId)
            .build();
    auto registry =
        memory::CustomMemoryResourceRegistry::createRegistry(nullptr);
    queryCtx->setRegistry<memory::CustomMemoryResourceRegistry::Registry>(
        memory::kCustomMemoryResourceRegistryKey, registry);
    registry->insert(std::string{CxlMemoryResource::kTag}, resource);
    return queryCtx;
  }

  // Sums the relocatedMemoryBytes runtime stat across the task's operators. The
  // stat is recorded in release builds, so the assertions hold in every build
  // mode.
  static int64_t relocatedBytes(const std::shared_ptr<exec::Task>& task) {
    int64_t total{0};
    for (const auto& [_, nodeStats] : exec::toPlanStats(task->taskStats())) {
      const auto it = nodeStats.customStats.find(
          std::string(memory::kRelocatedMemoryBytes));
      if (it != nodeStats.customStats.end()) {
        total += it->second.sum;
      }
    }
    return total;
  }

  // Drives an aggregation that fires reclaim on every one of many batches in
  // migrate mode, so the DRAM/tier accounting is exercised across many reclaims
  // (and rehashes) plus teardown. 'includeBuckets' also migrates the bucket
  // array. Asserts relocation fired and nothing spilled to disk; the base
  // class's pool leak checks assert the accounting balances.
  void runRepeatedMigration(const std::string& queryId, bool includeBuckets);

  // Returns whether move_pages works on this host, so the migrate-mode test can
  // skip where the syscall is unavailable. Migrating to the local node 0 is a
  // valid no-op migration everywhere else.
  static bool movePagesAvailable() {
    auto pool = memory::memoryManager()->addLeafPool("move-pages-probe");
    memory::AllocationPool probe(pool.get());
    *reinterpret_cast<int64_t*>(probe.allocateFixed(4096, 1)) = 1;
    try {
      probe.migratePagesToNode(0);
    } catch (const VeloxException&) {
      return false;
    }
    return true;
  }
};

// Under memory arbitration a grouped, fixed-width aggregation reclaims by
// relocating its groups to the configured tier rather than spilling to disk.
// Spill is enabled so relocation can ride the reclaim path, but the tier
// intercepts reclaim before any disk write. A null grouping key checks that the
// destination container's column stats are carried over, else the null group
// reads back as a non-null garbage key. Relocation must move bytes and no disk
// spill must occur (either/or with spill).
TEST_F(CxlAggregationRelocateTest, relocatesToTierUnderArbitration) {
  // Front-load the table: the large first batch populates the groups, the tiny
  // second batch only exists to trigger the second addInput where the injected
  // reclaim fires against a full table.
  constexpr int32_t kFirstBatchRows = 200'000;
  auto firstBatch = makeRowVector({
      // Distinct keys, plus a null key every 1000th row.
      makeFlatVector<int64_t>(
          kFirstBatchRows,
          [](auto row) { return static_cast<int64_t>(row); },
          [](auto row) { return row % 1'000 == 0; }),
      makeFlatVector<int64_t>(kFirstBatchRows, [](auto row) { return row; }),
  });
  auto secondBatch = makeRowVector({
      makeFlatVector<int64_t>(
          8, [](auto row) { return kFirstBatchRows + row; }),
      makeFlatVector<int64_t>(8, [](auto row) { return row; }),
  });
  std::vector<RowVectorPtr> batches{firstBatch, secondBatch};
  createDuckDbTable(batches);

  // Fire the reclaim path exactly once; reclaim() relocates to the tier instead
  // of spilling to disk. This drives a real arbitration, observed in release
  // builds too.
  exec::TestScopedSpillInjection injectSpill(
      /*spillPct=*/100, /*poolRegExp=*/".*", /*maxInjections=*/1);

  auto spillDirectory = exec::test::TempDirectoryPath::create();
  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(batches)
                        .singleAggregation({"c0"}, {"sum(c1)"})
                        .planNode();
  params.queryCtx = makeQueryCtxWithTier("cxl-agg-arbitrator");
  params.spillDirectory = spillDirectory->getPath();
  params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kRelocationResourceTag] =
      std::string{CxlMemoryResource::kTag};
  auto task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_GT(relocatedBytes(task), 0);
  // Either/or: relocation replaced disk spill, so nothing reached disk.
  for (const auto& [_, nodeStats] : exec::toPlanStats(task->taskStats())) {
    EXPECT_EQ(nodeStats.spilledBytes, 0);
  }
}

// move_pages relocation: under arbitration reclaim migrates the payload pages
// in place to a NUMA node (relocation_mode=migrate) instead of byte-copying,
// leaving virtual addresses and the hash index unchanged. Targets the local
// node 0 (a valid no-op migration on any Linux host). Migration must relieve
// the DRAM cap (relocatedMemoryBytes > 0), no disk spill must occur, results
// must be correct, and the teardown accounting must balance (checked by the
// base class's pool leak checks).
TEST_F(CxlAggregationRelocateTest, migratesToTierUnderArbitration) {
  if (!movePagesAvailable()) {
    GTEST_SKIP() << "move_pages not available on this host";
  }
  constexpr int32_t kFirstBatchRows = 200'000;
  auto firstBatch = makeRowVector({
      makeFlatVector<int64_t>(
          kFirstBatchRows,
          [](auto row) { return static_cast<int64_t>(row); },
          [](auto row) { return row % 1'000 == 0; }),
      makeFlatVector<int64_t>(kFirstBatchRows, [](auto row) { return row; }),
  });
  auto secondBatch = makeRowVector({
      makeFlatVector<int64_t>(
          8, [](auto row) { return kFirstBatchRows + row; }),
      makeFlatVector<int64_t>(8, [](auto row) { return row; }),
  });
  std::vector<RowVectorPtr> batches{firstBatch, secondBatch};
  createDuckDbTable(batches);

  exec::TestScopedSpillInjection injectSpill(
      /*spillPct=*/100, /*poolRegExp=*/".*", /*maxInjections=*/1);

  auto spillDirectory = exec::test::TempDirectoryPath::create();
  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(batches)
                        .singleAggregation({"c0"}, {"sum(c1)"})
                        .planNode();
  params.queryCtx = makeQueryCtxWithTier("cxl-agg-migrate");
  params.spillDirectory = spillDirectory->getPath();
  params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kRelocationResourceTag] =
      std::string{CxlMemoryResource::kTag};
  params.queryConfigs[core::QueryConfig::kRelocationMode] = "migrate";
  params.queryConfigs[core::QueryConfig::kRelocationMigrateNumaNode] = "0";
  auto task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_GT(relocatedBytes(task), 0);
  // Either/or: migration replaced disk spill, so nothing reached disk.
  for (const auto& [_, nodeStats] : exec::toPlanStats(task->taskStats())) {
    EXPECT_EQ(nodeStats.spilledBytes, 0);
  }
}

// Repeated reclaim in migrate mode: the benchmark's config D fires reclaim many
// times as the table grows under a tight DRAM cap, not once. Each reclaim
// migrates only the newly grown extent and adjusts the DRAM/tier accounting;
// after many reclaims plus teardown the base class's pool leak checks must
// still balance. Reproduces the config-D double-accounting failure.
void CxlAggregationRelocateTest::runRepeatedMigration(
    const std::string& queryId,
    bool includeBuckets) {
  // Many batches of distinct keys so the table grows across many addInput
  // calls, and reclaim fires on each. Row width (key + sum accumulator) makes
  // the used extent not page-aligned at reclaim time, exercising the
  // incremental resume.
  constexpr int32_t kNumBatches = 16;
  constexpr int32_t kBatchRows = 50'000;
  std::vector<RowVectorPtr> batches;
  batches.reserve(kNumBatches);
  for (int32_t batch = 0; batch < kNumBatches; ++batch) {
    const int64_t base = static_cast<int64_t>(batch) * kBatchRows;
    batches.push_back(makeRowVector({
        makeFlatVector<int64_t>(
            kBatchRows, [&](auto row) { return base + row; }),
        makeFlatVector<int64_t>(kBatchRows, [](auto row) { return row; }),
    }));
  }
  createDuckDbTable(batches);

  exec::TestScopedSpillInjection injectSpill(
      /*spillPct=*/100, /*poolRegExp=*/".*", /*maxInjections=*/kNumBatches);

  auto spillDirectory = exec::test::TempDirectoryPath::create();
  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(batches)
                        .singleAggregation({"c0"}, {"sum(c1)"})
                        .planNode();
  params.queryCtx = makeQueryCtxWithTier(queryId);
  params.spillDirectory = spillDirectory->getPath();
  params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kRelocationResourceTag] =
      std::string{CxlMemoryResource::kTag};
  params.queryConfigs[core::QueryConfig::kRelocationMode] = "migrate";
  params.queryConfigs[core::QueryConfig::kRelocationMigrateNumaNode] = "0";
  params.queryConfigs[core::QueryConfig::kRelocationMigrateIncludeBuckets] =
      includeBuckets ? "true" : "false";
  auto task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_GT(relocatedBytes(task), 0);
  for (const auto& [_, nodeStats] : exec::toPlanStats(task->taskStats())) {
    EXPECT_EQ(nodeStats.spilledBytes, 0);
  }
}

TEST_F(CxlAggregationRelocateTest, repeatedMigrationAccountingBalances) {
  if (!movePagesAvailable()) {
    GTEST_SKIP() << "move_pages not available on this host";
  }
  runRepeatedMigration("cxl-agg-migrate-repeated", /*includeBuckets=*/false);
}

// With --migrate_buckets the bucket array is migrated too. It is reallocated on
// rehash, so its bytes must be excluded from the relieved accounting --
// counting them would relieve pages that the rehash later frees,
// double-releasing them. Many reclaims interleave rehashes, so the accounting
// must still balance.
TEST_F(
    CxlAggregationRelocateTest,
    repeatedMigrationWithBucketsAccountingBalances) {
  if (!movePagesAvailable()) {
    GTEST_SKIP() << "move_pages not available on this host";
  }
  runRepeatedMigration("cxl-agg-migrate-buckets", /*includeBuckets=*/true);
}

// A migration whose CXL tier budget charge fails mid-reclaim must neither crash
// nor fail the query. reclaimByMigration() relieves the DRAM cap
// (reportExternalFree) and then charges the tier budget
// (reportExternalAllocation), which throws when the tier is exhausted -- what
// benchmark config D hits. On the buggy path the relief is applied but the
// bytes go untracked, the exception aborts the pool, and the payload's teardown
// free drives the leaf reservation below zero (a 2^64 unsigned wrap). The fix
// records the relief before the charge and treats the charge as best-effort
// (the pages already moved physically), so the query completes and teardown
// balances. Injected with a debug-only TestValue on the first migration.
TEST_F(CxlAggregationRelocateTest, migrateToleratesTierChargeFailure) {
  if (!movePagesAvailable()) {
    GTEST_SKIP() << "move_pages not available on this host";
  }
  constexpr int32_t kFirstBatchRows = 200'000;
  auto firstBatch = makeRowVector({
      makeFlatVector<int64_t>(
          kFirstBatchRows, [](auto row) { return static_cast<int64_t>(row); }),
      makeFlatVector<int64_t>(kFirstBatchRows, [](auto row) { return row; }),
  });
  auto secondBatch = makeRowVector({
      makeFlatVector<int64_t>(
          8, [](auto row) { return kFirstBatchRows + row; }),
      makeFlatVector<int64_t>(8, [](auto row) { return row; }),
  });
  std::vector<RowVectorPtr> batches{firstBatch, secondBatch};
  createDuckDbTable(batches);

  exec::TestScopedSpillInjection injectSpill(
      /*spillPct=*/100, /*poolRegExp=*/".*", /*maxInjections=*/1);

  // Fail the tier budget charge on the first migration, mimicking a full CXL
  // tier. reportExternalFree() has already relieved the DRAM cap when this
  // fires, so the relief must survive to be reversed at teardown.
  std::atomic_int migrations{0};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashAggregation::reclaimByMigration::tierCharge",
      std::function<void(void*)>([&](void*) {
        if (migrations.fetch_add(1) == 0) {
          VELOX_FAIL("injected CXL tier exhaustion");
        }
      }));

  auto spillDirectory = exec::test::TempDirectoryPath::create();
  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(batches)
                        .singleAggregation({"c0"}, {"sum(c1)"})
                        .planNode();
  params.queryCtx = makeQueryCtxWithTier("cxl-agg-tier-charge-fails");
  params.spillDirectory = spillDirectory->getPath();
  params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kRelocationResourceTag] =
      std::string{CxlMemoryResource::kTag};
  params.queryConfigs[core::QueryConfig::kRelocationMode] = "migrate";
  params.queryConfigs[core::QueryConfig::kRelocationMigrateNumaNode] = "0";
  // The query completes with correct results despite the injected tier
  // exhaustion; the base class's pool leak checks assert the teardown balance.
  auto task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");
  EXPECT_GT(relocatedBytes(task), 0);
}

// Relocation does not depend on disk spill. With only 'relocation_resource_tag'
// set -- spill disabled, no spill directory -- memory arbitration still
// reclaims by relocating the payload to the tier.
TEST_F(CxlAggregationRelocateTest, relocatesWithoutSpillEnabled) {
  constexpr int32_t kFirstBatchRows = 200'000;
  auto firstBatch = makeRowVector({
      makeFlatVector<int64_t>(
          kFirstBatchRows, [](auto row) { return static_cast<int64_t>(row); }),
      makeFlatVector<int64_t>(kFirstBatchRows, [](auto row) { return row; }),
  });
  auto secondBatch = makeRowVector({
      makeFlatVector<int64_t>(
          8, [](auto row) { return kFirstBatchRows + row; }),
      makeFlatVector<int64_t>(8, [](auto row) { return row; }),
  });
  std::vector<RowVectorPtr> batches{firstBatch, secondBatch};
  createDuckDbTable(batches);

  exec::TestScopedSpillInjection injectSpill(
      /*spillPct=*/100, /*poolRegExp=*/".*", /*maxInjections=*/1);

  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(batches)
                        .singleAggregation({"c0"}, {"sum(c1)"})
                        .planNode();
  params.queryCtx = makeQueryCtxWithTier("cxl-agg-no-spill");
  // Only relocation is configured; disk spill is left disabled.
  params.queryConfigs[core::QueryConfig::kRelocationResourceTag] =
      std::string{CxlMemoryResource::kTag};
  auto task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_GT(relocatedBytes(task), 0);
  for (const auto& [_, nodeStats] : exec::toPlanStats(task->taskStats())) {
    EXPECT_EQ(nodeStats.spilledBytes, 0);
  }
}

// A varchar grouping key uses external (out-of-line) memory, so its rows cannot
// be byte-relocated. The gate must exclude it: even with spill enabled and a
// tier configured, reclaim falls back to disk spill (which would otherwise
// throw in relocateRunsTo) and the query returns correct results.
TEST_F(CxlAggregationRelocateTest, varcharKeyDoesNotRelocate) {
  auto data = makeRowVector({
      makeFlatVector<std::string>(
          1'000, [](auto row) { return fmt::format("k{}", row % 17); }),
      makeFlatVector<int64_t>(1'000, [](auto row) { return row; }),
  });
  createDuckDbTable({data});

  exec::TestScopedSpillInjection injectSpill(
      /*spillPct=*/100, /*poolRegExp=*/".*", /*maxInjections=*/1);

  auto spillDirectory = exec::test::TempDirectoryPath::create();
  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values({data})
                        .singleAggregation({"c0"}, {"sum(c1)"})
                        .planNode();
  params.queryCtx = makeQueryCtxWithTier("cxl-agg-varchar");
  params.spillDirectory = spillDirectory->getPath();
  params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kRelocationResourceTag] =
      std::string{CxlMemoryResource::kTag};
  auto task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_EQ(relocatedBytes(task), 0);
}

// A partial aggregation is not a complete reclaim point, so the gate excludes
// it: relocation never fires and results are correct.
TEST_F(CxlAggregationRelocateTest, partialAggregationDoesNotRelocate) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>(1'000, [](auto row) { return row % 10; }),
      makeFlatVector<int64_t>(1'000, [](auto row) { return row; }),
  });
  createDuckDbTable({data});

  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values({data})
                        .partialAggregation({"c0"}, {"sum(c1)"})
                        .finalAggregation()
                        .planNode();
  params.queryCtx = makeQueryCtxWithTier("cxl-agg-partial");
  params.queryConfigs[core::QueryConfig::kRelocationResourceTag] =
      std::string{CxlMemoryResource::kTag};
  auto task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  // The partial stage never relocates; the final stage relocates only under
  // pressure, which this unpressured run does not create.
  EXPECT_EQ(relocatedBytes(task), 0);
}

// A pre-grouped (partially streaming) aggregation holds raw row pointers into
// the container across mid-stream flushes, so its payload cannot be relocated.
// The gate excludes it and reclaim spills to disk instead.
TEST_F(CxlAggregationRelocateTest, preGroupedKeyDoesNotRelocate) {
  // Input is clustered on c0 (runs of 100 equal values), so c0 can be
  // pre-grouped while c1 is not.
  auto data = makeRowVector({
      makeFlatVector<int64_t>(10'000, [](auto row) { return row / 100; }),
      makeFlatVector<int64_t>(10'000, [](auto row) { return row % 7; }),
      makeFlatVector<int64_t>(10'000, [](auto row) { return row; }),
  });
  createDuckDbTable({data});

  exec::TestScopedSpillInjection injectSpill(
      /*spillPct=*/100, /*poolRegExp=*/".*", /*maxInjections=*/1);

  auto spillDirectory = exec::test::TempDirectoryPath::create();
  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values({data})
                        .aggregation(
                            /*groupingKeys=*/{"c0", "c1"},
                            /*preGroupedKeys=*/{"c0"},
                            /*aggregates=*/{"sum(c2)"},
                            /*masks=*/{},
                            core::AggregationNode::Step::kSingle,
                            /*ignoreNullKeys=*/false)
                        .planNode();
  params.queryCtx = makeQueryCtxWithTier("cxl-agg-pregrouped");
  params.spillDirectory = spillDirectory->getPath();
  params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
  params.queryConfigs[core::QueryConfig::kRelocationResourceTag] =
      std::string{CxlMemoryResource::kTag};
  auto task =
      assertQuery(params, "SELECT c0, c1, sum(c2) FROM tmp GROUP BY c0, c1");

  EXPECT_EQ(relocatedBytes(task), 0);
}

} // namespace
} // namespace facebook::velox::cxl
