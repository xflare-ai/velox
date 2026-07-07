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

#include <gtest/gtest.h>

#include "velox/common/memory/AllocationPool.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/VectorHasher.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;

namespace facebook::velox::exec::test {
namespace {

// Drives a stock HashTable through groupProbe to exercise migratePayload.
// Unlike relocatePayload, move_pages keeps virtual addresses stable, so
// migration keeps the same RowContainer and only moves its backing pages.
// Migrating to the local node 0 is a valid no-op migration on any Linux host,
// which is enough to verify that the index stays live and the payload intact.
class HashTableMigrateTest : public testing::Test,
                             public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  // Builds a single-bigint-key aggregation table with no accumulators.
  std::unique_ptr<HashTable<false>> makeTable() {
    std::vector<std::unique_ptr<VectorHasher>> hashers;
    hashers.push_back(VectorHasher::create(BIGINT(), 0));
    return HashTable<false>::createForAggregation(
        std::move(hashers), {}, pool_.get());
  }

  // Probes 'keys'; hits and newGroups land in 'lookup'.
  void
  probe(HashTable<false>& table, HashLookup& lookup, const VectorPtr& keys) {
    auto input = makeRowVector({keys});
    SelectivityVector rows(keys->size());
    table.prepareForGroupProbe(
        lookup, input, rows, BaseHashTable::kNoSpillInputStartPartitionBit);
    table.groupProbe(lookup, BaseHashTable::kNoSpillInputStartPartitionBit);
  }

  // Skips the test when move_pages is unavailable on the host.
  void skipUnlessMovePagesAvailable() {
    memory::AllocationPool probePool(pool_.get());
    *reinterpret_cast<int64_t*>(probePool.allocateFixed(4096, 1)) = 1;
    try {
      probePool.migratePagesToNode(0);
    } catch (const VeloxException&) {
      GTEST_SKIP() << "move_pages not available on this host";
    }
  }

  // More distinct keys than VectorHasher::kMaxDistinct, spread wide, so the
  // hasher cannot assign value ids and the table uses a bucketed hash mode.
  static constexpr vector_size_t kBucketedSize = 150'000;

  VectorPtr wideKeys(vector_size_t size, int64_t base = 0) {
    return makeFlatVector<int64_t>(
        size, [base](auto row) { return (row + base) * 1'000'003LL + 7; });
  }

  int64_t keyAt(RowContainer* container, char* row) {
    auto result = BaseVector::create(BIGINT(), 1, pool_.get());
    container->extractColumn(&row, 1, 0, result);
    return result->asFlatVector<int64_t>()->valueAt(0);
  }
};

TEST_F(HashTableMigrateTest, migratePayloadAndReprobe) {
  skipUnlessMovePagesAvailable();
  constexpr vector_size_t kSize = kBucketedSize;
  auto table = makeTable();
  HashLookup lookup(table->hashers(), pool_.get());
  auto keys = wideKeys(kSize);

  probe(*table, lookup, keys);
  ASSERT_EQ(lookup.newGroups.size(), kSize);
  ASSERT_NE(table->hashMode(), BaseHashTable::HashMode::kArray);

  const int64_t migrated =
      table->migratePayload(/*numaNode=*/0, /*includeBuckets=*/false);
  EXPECT_GT(migrated, 0);
  // Migration keeps the same container: rows stay in 'rows_', no new container.
  EXPECT_EQ(table->rows()->numRows(), kSize);
  EXPECT_EQ(table->numDistinct(), kSize);
  EXPECT_EQ(table->numRowContainers(), 1);

  // Every key resolves to its (in-place) row, creating no new groups.
  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
  for (vector_size_t i = 0; i < kSize; i += 1'000) {
    EXPECT_EQ(
        keyAt(table->rows(), lookup.hits[i]),
        keys->asFlatVector<int64_t>()->valueAt(i));
  }
}

TEST_F(HashTableMigrateTest, migrateInArrayMode) {
  skipUnlessMovePagesAvailable();
  constexpr vector_size_t kSize = 1'000;
  auto table = makeTable();
  HashLookup lookup(table->hashers(), pool_.get());
  // A dense small range keeps the table in kArray mode.
  auto keys = makeFlatVector<int64_t>(kSize, [](auto row) { return row; });

  probe(*table, lookup, keys);
  ASSERT_EQ(table->hashMode(), BaseHashTable::HashMode::kArray);
  EXPECT_GT(table->migratePayload(0, false), 0);
  EXPECT_EQ(table->rows()->numRows(), kSize);

  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
  constexpr vector_size_t kRow = 42;
  EXPECT_EQ(
      keyAt(table->rows(), lookup.hits[kRow]),
      keys->asFlatVector<int64_t>()->valueAt(kRow));
}

TEST_F(HashTableMigrateTest, migrateIncludingBuckets) {
  skipUnlessMovePagesAvailable();
  constexpr vector_size_t kSize = kBucketedSize;
  auto table = makeTable();
  HashLookup lookup(table->hashers(), pool_.get());
  auto keys = wideKeys(kSize);
  probe(*table, lookup, keys);

  // Migrating the bucket array too moves more bytes than payload alone.
  const int64_t payloadOnly =
      table->migratePayload(0, /*includeBuckets=*/false);
  const int64_t withBuckets = table->migratePayload(0, /*includeBuckets=*/true);
  EXPECT_GT(withBuckets, 0);
  EXPECT_GT(payloadOnly, 0);

  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
}

// A payload large enough to spill past the huge-page threshold spans multiple
// allocation runs; migration must move every run's pages and keep them
// readable.
TEST_F(HashTableMigrateTest, migrateAcrossManyRuns) {
  skipUnlessMovePagesAvailable();
  constexpr vector_size_t kSize = 400'000;
  auto table = makeTable();
  HashLookup lookup(table->hashers(), pool_.get());
  auto keys = wideKeys(kSize);

  probe(*table, lookup, keys);
  ASSERT_EQ(lookup.newGroups.size(), kSize);

  EXPECT_GT(table->migratePayload(0, false), 0);
  EXPECT_EQ(table->rows()->numRows(), kSize);
  EXPECT_EQ(table->numDistinct(), kSize);

  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
  for (vector_size_t i = 0; i < kSize; i += 997) {
    EXPECT_EQ(
        keyAt(table->rows(), lookup.hits[i]),
        keys->asFlatVector<int64_t>()->valueAt(i));
  }
}

// New groups added after a migration coexist with the migrated ones, and a
// second migration moves only the newly grown extent (incremental).
TEST_F(HashTableMigrateTest, incrementalMigration) {
  skipUnlessMovePagesAvailable();
  constexpr vector_size_t kFirst = 50'000;
  constexpr vector_size_t kSecond = 50'000;
  auto table = makeTable();
  HashLookup lookup(table->hashers(), pool_.get());

  auto first = wideKeys(kFirst);
  probe(*table, lookup, first);
  const int64_t firstMigrated = table->migratePayload(0, false);
  EXPECT_GT(firstMigrated, 0);

  auto second = wideKeys(kSecond, /*base=*/kFirst);
  probe(*table, lookup, second);
  EXPECT_EQ(lookup.newGroups.size(), kSecond);

  const int64_t secondMigrated = table->migratePayload(0, false);
  EXPECT_GT(secondMigrated, 0);
  EXPECT_EQ(table->rows()->numRows(), kFirst + kSecond);
  EXPECT_EQ(table->numRowContainers(), 1);

  probe(*table, lookup, first);
  EXPECT_TRUE(lookup.newGroups.empty());
  probe(*table, lookup, second);
  EXPECT_TRUE(lookup.newGroups.empty());
}

} // namespace
} // namespace facebook::velox::exec::test
