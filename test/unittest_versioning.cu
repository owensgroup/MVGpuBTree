
/*
 *   Copyright 2022 The Regents of the University of California, Davis
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#include <gpu_btree.h>
#include <gtest/gtest.h>
#include <cmd.hpp>
#include <cstdint>

std::size_t num_keys_;
std::size_t average_range_length_;

namespace {
using key_type   = uint32_t;
using value_type = uint32_t;

const auto sentinel_value = std::numeric_limits<key_type>::max();
// const auto sentinel_key = std::numeric_limits<value_type>::max();
template <typename BTreeMap>
struct BTreeMapData {
  using btree_map = BTreeMap;
};

template <class MapData>
class VersionedBTreeMapTest : public testing::Test {
 protected:
  VersionedBTreeMapTest() { btree_map_ = new typename map_data::btree_map(); }
  ~VersionedBTreeMapTest() override { delete btree_map_; }
  using map_data = MapData;
  typename map_data::btree_map* btree_map_;
};

template <typename T>
struct mapped_vector {
  mapped_vector(std::size_t capacity) : capacity_(capacity) { allocate(capacity); }
  T& operator[](std::size_t index) { return dh_buffer_[index]; }
  ~mapped_vector() {}
  void free() {
    cuda_try(cudaDeviceSynchronize());
    cuda_try(cudaFreeHost(dh_buffer_));
  }
  T* data() const { return dh_buffer_; }

  std::vector<T> to_std_vector() {
    std::vector<T> copy(capacity_);
    for (std::size_t i = 0; i < capacity_; i++) { copy[i] = dh_buffer_[i]; }
    return copy;
  }
  std::size_t size() const { return capacity_; }

 private:
  void allocate(std::size_t count) { cuda_try(cudaMallocHost(&dh_buffer_, sizeof(T) * count)); }
  std::size_t capacity_;
  T* dh_buffer_;
};

struct testing_input {
  testing_input(std::size_t input_num_keys, key_type average_range_length = 32)
      : num_keys(input_num_keys)
      , average_rq_length(average_range_length)
      , keys(input_num_keys)
      , values(input_num_keys)
      , keys_exist(input_num_keys)
      , keys_not_exist(input_num_keys)
      , keys_upper_bound(input_num_keys)
      , keys_lower_bound(input_num_keys) {
    make_input();
  }
  void make_input() {
    for (std::size_t i = 0; i < num_keys; i++) {
      // Make sure that the input doesn't contain 0
      // and, queries that do not exist in the table are uniformly distributed to avoid
      // contention during deletion... an optimzation is to avoid locking if key doesn't
      // exist in node
      keys[i]             = static_cast<key_type>(i + 1) * 2;
      values[i]           = to_value(keys[i]);
      keys_exist[i]       = keys[i];
      keys_not_exist[i]   = keys[i] + 1;
      keys_lower_bound[i] = keys[i];
      keys_upper_bound[i] = keys_lower_bound[i] + average_rq_length * 2;
    }
  }
  value_type to_value(const key_type k) const { return static_cast<value_type>(k); }
  void free() {
    keys.free();
    values.free();
    keys_exist.free();
    keys_not_exist.free();
    keys_upper_bound.free();
    keys_lower_bound.free();
  }

  std::size_t num_keys;
  key_type average_rq_length;
  mapped_vector<key_type> keys;
  mapped_vector<value_type> values;
  mapped_vector<key_type> keys_exist;
  mapped_vector<key_type> keys_not_exist;
  mapped_vector<key_type> keys_upper_bound;
  mapped_vector<key_type> keys_lower_bound;
};

struct TreeParam {
  static constexpr int BranchingFactor = 16;
};
struct SlabAllocParam {
  static constexpr uint32_t NumSuperBlocks  = 4;
  static constexpr uint32_t NumMemoryBlocks = 1024 * 8;
  static constexpr uint32_t TileSize        = TreeParam::BranchingFactor;
  static constexpr uint32_t SlabSize        = 128;
};
using node_type           = GpuBTree::node_type<key_type, value_type, TreeParam::BranchingFactor>;
using bump_allocator_type = device_bump_allocator<node_type>;
using slab_allocator_type = device_allocator::SlabAllocLight<node_type,
                                                             SlabAllocParam::NumSuperBlocks,
                                                             SlabAllocParam::NumMemoryBlocks,
                                                             SlabAllocParam::TileSize,
                                                             SlabAllocParam::SlabSize>;

typedef testing::Types<BTreeMapData<GpuBTree::gpu_versioned_btree<key_type,
                                                                  value_type,
                                                                  TreeParam::BranchingFactor,
                                                                  bump_allocator_type>>,
                       BTreeMapData<GpuBTree::gpu_versioned_btree<key_type,
                                                                  value_type,
                                                                  TreeParam::BranchingFactor,
                                                                  slab_allocator_type>>>
    Implementations;

TYPED_TEST_SUITE(VersionedBTreeMapTest, Implementations);

TYPED_TEST(VersionedBTreeMapTest, InsertInPlaceValidation) {
  testing_input input(num_keys_);
  auto first_batch_size  = num_keys_ / 2;
  auto second_batch_size = num_keys_ - first_batch_size;
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // First batch
  this->btree_map_->insert(input.keys.data(), input.values.data(), first_batch_size);
  this->btree_map_->take_snapshot();
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto keys = input.keys.to_std_vector();
  keys.resize(first_batch_size);
  EXPECT_NO_THROW(this->btree_map_->validate_tree_structure(
      keys, [](auto key) { return static_cast<value_type>(key); }));

  // Second batch
  this->btree_map_->insert(input.keys.data() + first_batch_size,
                           input.values.data() + first_batch_size,
                           second_batch_size);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  keys = input.keys.to_std_vector();
  EXPECT_NO_THROW(this->btree_map_->validate_tree_structure(
      keys, [](auto key) { return static_cast<value_type>(key); }));

  input.free();
}

TYPED_TEST(VersionedBTreeMapTest, InsertOutOfPlaceValidation) {
  testing_input input(num_keys_);
  auto first_batch_size  = num_keys_ / 2;
  auto second_batch_size = num_keys_ - first_batch_size;
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // First batch
  this->btree_map_->insert(input.keys.data(), input.values.data(), first_batch_size, 0, false);
  this->btree_map_->take_snapshot();
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto keys = input.keys.to_std_vector();
  keys.resize(first_batch_size);
  EXPECT_NO_THROW(this->btree_map_->validate_tree_structure(
      keys, [](auto key) { return static_cast<value_type>(key); }));

  // Second batch
  this->btree_map_->insert(input.keys.data() + first_batch_size,
                           input.values.data() + first_batch_size,
                           second_batch_size,
                           0,
                           false);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  keys = input.keys.to_std_vector();
  EXPECT_NO_THROW(this->btree_map_->validate_tree_structure(
      keys, [](auto key) { return static_cast<value_type>(key); }));

  input.free();
}

TYPED_TEST(VersionedBTreeMapTest, FindExist) {
  testing_input input(num_keys_);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto first_batch_size  = num_keys_ / 2;
  auto second_batch_size = num_keys_ - first_batch_size;
  mapped_vector<value_type> find_results(num_keys_);

  // First batch
  this->btree_map_->insert(input.keys.data(), input.values.data(), first_batch_size, 0, false);
  auto snapshot_id = this->btree_map_->take_snapshot();
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  this->btree_map_->find(
      input.keys_exist.data(), find_results.data(), first_batch_size, snapshot_id);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < first_batch_size; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }

  // Second batch
  this->btree_map_->insert(input.keys.data() + first_batch_size,
                           input.values.data() + first_batch_size,
                           second_batch_size,
                           0,
                           false);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  this->btree_map_->find(input.keys_exist.data(), find_results.data(), num_keys_);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys_; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }

  input.free();
  find_results.free();
}

TYPED_TEST(VersionedBTreeMapTest, FindNotExist) {
  testing_input input(num_keys_);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto first_batch_size  = num_keys_ / 2;
  auto second_batch_size = num_keys_ - first_batch_size;
  mapped_vector<value_type> find_results(num_keys_);

  // First batch
  this->btree_map_->insert(input.keys.data(), input.values.data(), first_batch_size, 0, false);
  auto snapshot_id = this->btree_map_->take_snapshot();
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  this->btree_map_->find(input.keys_exist.data() + first_batch_size,
                         find_results.data(),
                         second_batch_size,
                         snapshot_id);

  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < second_batch_size; i++) {
    auto expected_value = sentinel_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }

  // Second batch
  this->btree_map_->insert(input.keys.data() + first_batch_size,
                           input.values.data() + first_batch_size,
                           second_batch_size,
                           0,
                           false);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  this->btree_map_->find(input.keys_not_exist.data(), find_results.data(), num_keys_);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys_; i++) {
    auto expected_value = sentinel_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }

  input.free();
  find_results.free();
}

TYPED_TEST(VersionedBTreeMapTest, RangeQueryTest) {
  const uint32_t rq_length = average_range_length_;
  const uint32_t num_rqs   = num_keys_;

  testing_input input(num_keys_, rq_length);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto first_batch_size  = num_keys_ / 2;
  auto second_batch_size = num_keys_ - first_batch_size;

  using pair_type = pair_type<key_type, value_type>;
  mapped_vector<pair_type> rq_results(rq_length * num_rqs);

  // First batch
  this->btree_map_->insert(input.keys.data(), input.values.data(), first_batch_size, 0, false);
  auto snapshot_id = this->btree_map_->take_snapshot();
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  this->btree_map_->range_query(input.keys_lower_bound.data(),
                                input.keys_upper_bound.data(),
                                rq_results.data(),
                                nullptr,
                                rq_length,
                                first_batch_size,
                                snapshot_id);

  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // validate
  std::set<key_type> ref_set_sc0(input.keys.data(), input.keys.data() + first_batch_size);
  for (std::size_t i = 0; i < first_batch_size; i++) {
    key_type query_lower_bound = input.keys_lower_bound[i];
    key_type query_upper_bound = input.keys_upper_bound[i];

    auto lower_bound               = ref_set_sc0.lower_bound(query_lower_bound);
    std::size_t range_start_offset = i * rq_length;
    std::size_t result_offset      = range_start_offset;
    while (lower_bound != ref_set_sc0.end() && (*lower_bound) >= query_lower_bound &&
           (*lower_bound) < query_upper_bound) {
      auto expected_key   = *lower_bound;
      auto expected_value = input.to_value(expected_key);
      auto expected_pair  = pair_type(expected_key, expected_value);
      auto found_pair     = rq_results[result_offset];
      ASSERT_EQ(expected_pair.first, found_pair.first);
      ASSERT_EQ(expected_pair.second, found_pair.second);
      lower_bound++;
      result_offset++;
    }
  }

  // Second batch
  this->btree_map_->insert(input.keys.data() + first_batch_size,
                           input.values.data() + first_batch_size,
                           second_batch_size,
                           0,
                           false);
  snapshot_id = this->btree_map_->take_snapshot();
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  this->btree_map_->range_query(input.keys_lower_bound.data(),
                                input.keys_upper_bound.data(),
                                rq_results.data(),
                                nullptr,
                                rq_length,
                                num_keys_,
                                snapshot_id);

  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::set<key_type> ref_set_sc1(input.keys.data(), input.keys.data() + input.keys.size());
  for (std::size_t i = 0; i < input.keys_lower_bound.size(); i++) {
    key_type query_lower_bound = input.keys_lower_bound[i];
    key_type query_upper_bound = input.keys_upper_bound[i];

    auto lower_bound               = ref_set_sc1.lower_bound(query_lower_bound);
    std::size_t range_start_offset = i * rq_length;
    std::size_t result_offset      = range_start_offset;
    while (lower_bound != ref_set_sc1.end() && (*lower_bound) >= query_lower_bound &&
           (*lower_bound) < query_upper_bound) {
      auto expected_key   = *lower_bound;
      auto expected_value = input.to_value(expected_key);
      auto expected_pair  = pair_type(expected_key, expected_value);
      auto found_pair     = rq_results[result_offset];
      ASSERT_EQ(expected_pair.first, found_pair.first);
      ASSERT_EQ(expected_pair.second, found_pair.second);
      lower_bound++;
      result_offset++;
    }
  }
  input.free();
  rq_results.free();
}

// TYPED_TEST(VersionedBTreeMapTest, ConcurrentInsertRangeQueryTest) {
//   const uint32_t rq_length = average_range_length_;
//   const uint32_t num_rqs = num_keys_;

//   testing_input input(num_keys_, rq_length);
//   EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

//   using pair_type = pair_type<key_type, value_type>;
//   mapped_vector<pair_type> rq_results(rq_length * num_rqs);

//   // First batch
//   this->btree_map_->concurrent_insert_range(input.keys.data(),
//                                             input.values.data(),
//                                             num_keys_,
//                                             input.keys_lower_bound.data(),
//                                             input.keys_upper_bound.data(),
//                                             num_keys_,
//                                             rq_results.data(),
//                                             nullptr,
//                                             rq_length);
//   EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

//   // validate
//   std::set<key_type> ref_set_sc0(input.keys.data(), input.keys.data() +
//   first_batch_size); for (std::size_t i = 0; i < first_batch_size; i++) {
//     key_type query_lower_bound = input.keys_lower_bound[i];
//     key_type query_upper_bound = input.keys_upper_bound[i];

//     auto lower_bound = ref_set_sc0.lower_bound(query_lower_bound);
//     std::size_t range_start_offset = i * rq_length;
//     std::size_t result_offset = range_start_offset;
//     while (lower_bound != ref_set_sc0.end() && (*lower_bound) >= query_lower_bound &&
//            (*lower_bound) < query_upper_bound) {
//       auto expected_key = *lower_bound;
//       auto expected_value = input.to_value(expected_key);
//       auto expected_pair = pair_type(expected_key, expected_value);
//       auto found_pair = rq_results[result_offset];
//       ASSERT_EQ(expected_pair.first, found_pair.first);
//       ASSERT_EQ(expected_pair.second, found_pair.second);
//       lower_bound++;
//       result_offset++;
//     }
//   }

//   // Second batch
//   this->btree_map_->insert(input.keys.data() + first_batch_size,
//                            input.values.data() + first_batch_size,
//                            second_batch_size,
//                            0,
//                            false);
//   snapshot_id = this->btree_map_->take_snapshot();
//   EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

//   this->btree_map_->range_query(input.keys_lower_bound.data(),
//                                 input.keys_upper_bound.data(),
//                                 rq_results.data(),
//                                 nullptr,
//                                 rq_length,
//                                 num_keys_,
//                                 snapshot_id);

//   EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
//   std::set<key_type> ref_set_sc1(input.keys.data(),
//                                  input.keys.data() + input.keys.size());
//   for (std::size_t i = 0; i < input.keys_lower_bound.size(); i++) {
//     key_type query_lower_bound = input.keys_lower_bound[i];
//     key_type query_upper_bound = input.keys_upper_bound[i];

//     auto lower_bound = ref_set_sc1.lower_bound(query_lower_bound);
//     std::size_t range_start_offset = i * rq_length;
//     std::size_t result_offset = range_start_offset;
//     while (lower_bound != ref_set_sc1.end() && (*lower_bound) >= query_lower_bound &&
//            (*lower_bound) < query_upper_bound) {
//       auto expected_key = *lower_bound;
//       auto expected_value = input.to_value(expected_key);
//       auto expected_pair = pair_type(expected_key, expected_value);
//       auto found_pair = rq_results[result_offset];
//       ASSERT_EQ(expected_pair.first, found_pair.first);
//       ASSERT_EQ(expected_pair.second, found_pair.second);
//       lower_bound++;
//       result_offset++;
//     }
//   }
//   input.free();
//   rq_results.free();
// }

}  // namespace

int main(int argc, char** argv) {
  auto arguments        = std::vector<std::string>(argv, argv + argc);
  num_keys_             = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1024);
  average_range_length_ = get_arg_value<uint32_t>(arguments, "range-length").value_or(2);
  std::cout << "Testing using:\n";
  std::cout << "Num keys = " << num_keys_ << '\n';
  std::cout << "Range query length = " << average_range_length_ << '\n';
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}