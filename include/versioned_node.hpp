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

#pragma once
#include <cstddef>
#include <cstdint>
#include <macros.hpp>
#include <type_traits>
#include <utils.hpp>

// #define DEBUG_LOCKING
// #define DEBUG_CHECKS
template <typename pair_type, typename tile_type, int node_width = 16>
struct btree_versioned_node {
  using key_type   = typename pair_type::key_type;
  using value_type = typename pair_type::value_type;
  using size_type  = uint32_t;
  using unsigned_type =
      typename std::conditional<sizeof(key_type) == sizeof(uint32_t), uint32_t, uint64_t>::type;

  DEVICE_QUALIFIER btree_versioned_node(pair_type* ptr, const tile_type& tile)
      : node_ptr_(ptr), tile_(tile), is_locked_(false) {}
  DEVICE_QUALIFIER btree_versioned_node(pair_type* ptr,
                                        const tile_type& tile,
                                        const pair_type pair,
                                        bool is_locked,
                                        bool is_intermediate)
      : node_ptr_(ptr)
      , lane_pair_(pair)
      , tile_(tile)
      , is_locked_(is_locked)
      , is_intermediate_(is_intermediate) {}

  DEVICE_QUALIFIER void store_unlocked_copy_at(pair_type* ptr) {
    auto to_store = lane_pair_;
    if (tile_.thread_rank() == metadata_lane_) { to_store.second = mask_lock_bit(to_store.second); }
    cuda_memory<pair_type>::store(
        ptr + tile_.thread_rank(), to_store, cuda_memory_order::memory_order_relaxed);
  }

  DEVICE_QUALIFIER void load(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    lane_pair_       = cuda_memory<pair_type>::load(node_ptr_ + tile_.thread_rank(), order);
    is_intermediate_ = !(get_sibling_data() & leaf_bit_mask_);
    is_locked_       = (get_sibling_data() & lock_bit_mask_);
  }
  DEVICE_QUALIFIER void store(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    cuda_memory<pair_type>::store(node_ptr_ + tile_.thread_rank(), lane_pair_, order);
    is_intermediate_ = !(get_sibling_data() & leaf_bit_mask_);
  }

  DEVICE_QUALIFIER void print() const {
    printf("node: %p, rank: %i, pair{%i, %i}\n",
           node_ptr_,
           tile_.thread_rank(),
           lane_pair_.first,
           lane_pair_.second);
  }

  DEVICE_QUALIFIER void print(uint32_t lookup_ts) const {
    printf("node: %p, (looking for %i) rank: %i, pair{%i, %i}\n",
           node_ptr_,
           lookup_ts,
           tile_.thread_rank(),
           lane_pair_.first,
           lane_pair_.second);
  }
  DEVICE_QUALIFIER void print_hex() const {
    printf("node: %p, rank: %i, pair{%#010x, %#010x}\n",
           node_ptr_,
           tile_.thread_rank(),
           lane_pair_.first,
           lane_pair_.second);
  }

  template <typename size_type>
  DEVICE_QUALIFIER btree_versioned_node do_split(const size_type right_sibling_index,
                                                 pair_type* right_sibling_ptr,
                                                 const bool make_sibling_locked = false,
                                                 const bool is_intermediate     = true) {
    // find the two minimum keys
    auto sibling_minimum = get_key_from_lane(node_width >> 1);
    // prepare the upper half in right sibling
    auto upper_key   = tile_.shfl_down(lane_pair_.first, half_node_width_);
    auto upper_value = tile_.shfl_down(lane_pair_.second, half_node_width_);

    // store sibling information
    auto this_node_sibling_highkey = get_key_from_lane(metadata_lane_);
    auto this_node_sibling_index   = get_value_from_lane(metadata_lane_);

    auto this_node_ts             = get_key_from_lane(version_lane_);
    auto this_node__next_ts_index = get_value_from_lane(version_lane_);

    // overwrite upper half-node
    pair_type upper_pair;
    if (tile_.thread_rank() >= (half_node_width_ - 2)) {
      upper_pair = pair_type();
    } else {
      upper_pair = pair_type(upper_key, upper_value);
    }
    if (tile_.thread_rank() >= half_node_width_) { lane_pair_ = pair_type(); }

    if (tile_.thread_rank() == metadata_lane_) {
      auto sibling_sibling_index   = mask_meta_bit(this_node_sibling_index);
      auto sibling_sibling_highkey = this_node_sibling_highkey;

      this_node_sibling_index   = right_sibling_index;
      this_node_sibling_highkey = sibling_minimum;

      if (is_locked_) { this_node_sibling_index = set_lock_bit(this_node_sibling_index); }
      if (make_sibling_locked) { sibling_sibling_index = set_lock_bit(sibling_sibling_index); }
      if (!is_intermediate_) {
        sibling_sibling_index   = set_leaf_bit(sibling_sibling_index);
        this_node_sibling_index = set_leaf_bit(this_node_sibling_index);
      }
      lane_pair_ = pair_type(this_node_sibling_highkey, this_node_sibling_index);
      upper_pair = pair_type(sibling_sibling_highkey, sibling_sibling_index);
    }
    if (tile_.thread_rank() == version_lane_ && is_intermediate) {
      lane_pair_ = {this_node_ts, this_node__next_ts_index};
      upper_pair = {this_node_ts, this_node__next_ts_index};
    } else if (tile_.thread_rank() == version_lane_ && !is_intermediate) {
      lane_pair_       = pair_type();
      upper_pair       = pair_type();
      lane_pair_.first = this_node_ts;
      upper_pair.first = this_node_ts;
    }
    return btree_versioned_node(
        right_sibling_ptr, tile_, upper_pair, make_sibling_locked, is_intermediate_);
  }

  struct split_intermediate_result {
    btree_versioned_node parent;
    btree_versioned_node sibling;
  };
  template <typename size_type>
  DEVICE_QUALIFIER split_intermediate_result split(const size_type right_sibling_index,
                                                   const size_type parent_index,
                                                   pair_type* right_sibling_ptr,
                                                   pair_type* parent_ptr,
                                                   const bool make_sibling_locked = false) {
    // We assume here that the parent is locked
    auto split_result = do_split(right_sibling_index, right_sibling_ptr, make_sibling_locked);

    // Update parent
    auto parent_node = btree_versioned_node(parent_ptr, tile_);
    parent_node.load(cuda_memory_order::memory_order_relaxed);

#ifdef DEBUG_CHECKS
    // Debug: parent must be locked and not full here
    bool parent_is_locked = parent_node.is_locked();
    cuda_assert(parent_is_locked);
    bool parent_is_full = parent_node.is_full();
    cuda_assert(!parent_is_full);
#endif
    // update the parent
    auto pivot_key = split_result.get_key_from_lane(0);
#ifdef DEBUG_CHECKS
    bool parent_updated = parent_node.insert(pivot_key, right_sibling_index);
    cuda_assert(parent_updated);
#else
    parent_node.insert(pivot_key, right_sibling_index);
#endif

    return {parent_node, split_result};
  }

  struct two_nodes_result {
    btree_versioned_node left;
    btree_versioned_node right;
  };
  template <typename size_type>
  DEVICE_QUALIFIER two_nodes_result split_as_root(const size_type left_sibling_index,
                                                  const size_type right_sibling_index,
                                                  pair_type* left_sibling_ptr,
                                                  pair_type* right_sibling_ptr,
                                                  const bool make_children_locked = false) {
    // Create a new root
    auto right_node_minimum = get_key_from_lane(node_width >> 1);

    // Copy the current node into a child
    auto left_child = btree_versioned_node(
        left_sibling_ptr, tile_, lane_pair_, make_children_locked, is_intermediate_);
    // if the root was a leaf, now it should be intermediate
    if (!is_intermediate_) { unset_leaf_in_registers(); }
    // Make new root
    if (tile_.thread_rank() == 0) {
      lane_pair_ = {lane_pair_.first, left_sibling_index};
    } else if (tile_.thread_rank() == 1) {
      lane_pair_ = {right_node_minimum, right_sibling_index};
    } else if (tile_.thread_rank() != metadata_lane_ && tile_.thread_rank() != version_lane_) {
      lane_pair_ = pair_type();
    }

    // now split the left child
    auto right_child =
        left_child.do_split(right_sibling_index, right_sibling_ptr, make_children_locked, false);

    return {left_child, right_child};
  }

  DEVICE_QUALIFIER int find_next_lane(const key_type& key) const {
    const bool valid_key              = lane_pair_.first != std::numeric_limits<uint32_t>::max();
    auto valid_lane                   = lane_is_valid();
    const bool key_greater_equal      = (key >= lane_pair_.first) && valid_key && valid_lane;
    uint32_t key_greater_equal_bitmap = tile_.ballot(key_greater_equal);
    auto next_lane                    = utils::bits::bfind(key_greater_equal_bitmap);

#ifdef DEBUG_CHECKS
    // This means that the key was smaller than everything else in the node which means traversal
    // was wrong
    cuda_assert(next_lane != -1);
#endif
    return next_lane;
  }

  DEVICE_QUALIFIER bool key_is_in_upperhalf(const key_type& key) const {
    auto next_lane = find_next_lane(key);
    return next_lane >= half_node_width_;
  }
  DEVICE_QUALIFIER size_type find_next(const key_type& key) const {
    auto next_lane = find_next_lane(key);
    return tile_.shfl(lane_pair_.second, next_lane);
  }

  DEVICE_QUALIFIER bool lane_is_valid() const {
    auto lane_id       = tile_.thread_rank();
    bool is_valid_lane = (lane_id != metadata_lane_) && (lane_id != version_lane_);
    return is_valid_lane;
  }
  DEVICE_QUALIFIER int find_key_lane_in_node(const key_type& key) const {
    bool is_valid_lane = lane_is_valid();
    auto key_exist     = tile_.ballot(lane_pair_.first == key && is_valid_lane);
    auto found_lane    = __ffs(key_exist);
    return found_lane - 1;
  }

  DEVICE_QUALIFIER int find_value_lane_in_node(const value_type& value) const {
    bool is_valid_lane = lane_is_valid();
    auto key_exist     = tile_.ballot(lane_pair_.second == value && is_valid_lane);
    auto found_lane    = __ffs(key_exist);
    return found_lane - 1;
  }

  DEVICE_QUALIFIER bool key_is_in_node(const key_type& key) const {
    auto key_location = find_key_lane_in_node(key);
    return key_location == -1 ? false : true;
  }

  DEVICE_QUALIFIER bool ptr_is_in_node(const size_type& ptr) const {
    auto ptr_location = find_value_lane_in_node(ptr);
    return ptr_location == -1 ? false : true;
  }

  DEVICE_QUALIFIER value_type get_key_value_from_node(const key_type& key) const {
    auto key_location = find_key_lane_in_node(key);
    return key_location == -1 ? std::numeric_limits<uint32_t>::max()
                              : get_value_from_lane(key_location);
  }
  DEVICE_QUALIFIER key_type get_key_from_lane(const int& location) const {
    return tile_.shfl(lane_pair_.first, location);
  }
  DEVICE_QUALIFIER void set_key_at_lane(const int& location, key_type key) {
    if (tile_.thread_rank() == location) { lane_pair_.first = key; }
  }
  DEVICE_QUALIFIER pair_type get_pair_from_lane(const int& location) const {
    auto key   = get_key_from_lane(location);
    auto value = get_value_from_lane(location);
    return pair_type(key, value);
  }
  DEVICE_QUALIFIER value_type get_value_from_lane(const int& location) const {
    return tile_.shfl(lane_pair_.second, location);
  }
  DEVICE_QUALIFIER void set_value_at_lane(const int& location, value_type val) {
    if (tile_.thread_rank() == location) { lane_pair_.second = val; }
  }
  DEVICE_QUALIFIER void set_pair_at_lane(const int& location, pair_type pair) {
    if (tile_.thread_rank() == location) { lane_pair_ = pair; }
  }
  DEVICE_QUALIFIER bool insert(const key_type key, const value_type value) {
#ifdef DEBUG_CHECKS
    // Debug: check if key is larger than high key
    auto high_key = get_high_key();
    cuda_assert(key < high_key);
#endif
    // check if key exist
    auto key_lane  = find_key_lane_in_node(key);
    bool key_exist = key_lane != -1;
    // if the key exists, we update the value
    if (key_exist) {
      if (tile_.thread_rank() == key_lane) { lane_pair_.second = value; }
      return false;
    } else {
      // else we shuffle the keys and do the insertion
      auto up_keys                  = tile_.shfl_up(lane_pair_.first, 1);
      auto up_values                = tile_.shfl_up(lane_pair_.second, 1);
      const bool valid_key          = lane_pair_.first != std::numeric_limits<uint32_t>::max();
      auto valid_lane               = lane_is_valid();
      const bool key_is_larger      = (key > lane_pair_.first) && valid_key && valid_lane;
      uint32_t key_is_larger_bitmap = tile_.ballot(key_is_larger);
      auto key_lane                 = utils::bits::bfind(key_is_larger_bitmap) + 1;
      if (tile_.thread_rank() < key_lane || !valid_lane) {
      } else if (tile_.thread_rank() == key_lane) {
        lane_pair_ = {key, value};
      } else {
        lane_pair_ = {up_keys, up_values};
      }
    }
    return true;
  }

  DEVICE_QUALIFIER bool erase(const key_type key) {
#ifdef DEBUG_CHECKS
    // Debug: check if key is larger than high key
    auto high_key = get_high_key();
    cuda_assert(key < high_key);
#endif
    // check if key exist
    auto key_lane  = find_key_lane_in_node(key);
    bool key_exist = key_lane != -1;

    if (key_exist) {
      auto down_keys   = tile_.shfl_down(lane_pair_.first, 1);
      auto down_values = tile_.shfl_down(lane_pair_.second, 1);
      if (tile_.thread_rank() >= key_lane && tile_.thread_rank() != metadata_lane_ &&
          tile_.thread_rank() != version_lane_) {
        lane_pair_ = {down_keys, down_values};
      }
      if (tile_.thread_rank() == last_pair_lane_) { lane_pair_ = pair_type(); }
    } else {
      return false;
    }
    return true;
  }

  // [lower_bound, upper_bound)
  DEVICE_QUALIFIER size_type get_in_range(const key_type& lower_bound,
                                          const key_type& upper_bound,
                                          pair_type* output) const {
    bool is_valid_lane   = lane_is_valid();
    bool in_range        = lane_pair_.first >= lower_bound && lane_pair_.first < upper_bound;
    auto in_range_ballot = tile_.ballot(in_range && is_valid_lane);
    auto first_lane      = __ffs(in_range_ballot) - 1;
    if (output != nullptr && is_valid_lane && in_range) {
#ifdef DEBUG_CHECKS
      cuda_assert(tile_.thread_rank() >= first_lane);
#endif
      output[tile_.thread_rank() - first_lane] = lane_pair_;
    }
    return __popc(in_range_ballot);
  }
  template <typename T>
  DEVICE_QUALIFIER T mask_meta_bit(const T& data) const {
    auto mask = (~lock_bit_mask_) & (~leaf_bit_mask_);
    return mask & data;
  }
  template <typename T>
  DEVICE_QUALIFIER T mask_lock_bit(const T& data) const {
    auto mask = (~lock_bit_mask_);
    return mask & data;
  }
  template <typename T>
  DEVICE_QUALIFIER T mask_leaf_bit(const T& data) const {
    auto mask = (~leaf_bit_mask_);
    return mask & data;
  }
  template <typename T>
  DEVICE_QUALIFIER T set_meta_bits(const T& data) const {
    auto mask = lock_bit_mask_ | leaf_bit_mask_;
    return mask | data;
  }
  template <typename T>
  DEVICE_QUALIFIER T set_lock_bit(const T& data) const {
    auto mask = lock_bit_mask_;
    return mask | data;
  }
  template <typename T>
  DEVICE_QUALIFIER T set_leaf_bit(const T& data) const {
    auto mask = leaf_bit_mask_;
    return mask | data;
  }
  template <typename T>
  DEVICE_QUALIFIER T unset_lock_bit(const T& data) const {
    auto mask = ~lock_bit_mask_;
    return mask & data;
  }
  template <typename T>
  DEVICE_QUALIFIER T unset_leaf_bit(const T& data) const {
    auto mask = ~leaf_bit_mask_;
    return mask & data;
  }
  DEVICE_QUALIFIER size_type get_sibling_data() const {
    // sibling is stored as value
    // it also has the two bits
    auto sibling_id_value = get_value_from_lane(metadata_lane_);
    auto sibling_id       = *reinterpret_cast<size_type*>(&sibling_id_value);
    return sibling_id;
  }

  DEVICE_QUALIFIER size_type get_sibling_index() const {
    // sibling is stored as value
    // it also has the two bits
    auto sibling_id_value = mask_meta_bit(get_value_from_lane(metadata_lane_));
    auto sibling_id       = *reinterpret_cast<size_type*>(&sibling_id_value);
    return sibling_id;
  }

  DEVICE_QUALIFIER key_type get_high_key() const {
    // highkey is stored as key
    auto high_key_value = get_key_from_lane(metadata_lane_);
    auto high_key       = *reinterpret_cast<key_type*>(&high_key_value);
    return high_key;
  }

  DEVICE_QUALIFIER size_type get_next_version_index() const {
    // sibling is stored as value
    auto next_version_id_value = get_value_from_lane(version_lane_);
    auto next_version_id       = *reinterpret_cast<size_type*>(&next_version_id_value);
    return next_version_id;
  }

  DEVICE_QUALIFIER key_type get_version_number() const {
    // highkey is stored as key
    auto version_value = get_key_from_lane(version_lane_);
    auto version       = *reinterpret_cast<key_type*>(&version_value);
    return version;
  }

  DEVICE_QUALIFIER void set_version_ptr_data(size_type ts, size_type ptr) {
    set_pair_at_lane(version_lane_, {ts, ptr});
  }

  DEVICE_QUALIFIER void set_timestamp(size_type ts) { set_key_at_lane(version_lane_, ts); }

  struct init_result {
    bool success;
    size_type cur_ts;
  };
  template <typename camera_type>
  DEVICE_QUALIFIER init_result init(const size_type invalid_ts, camera_type& camera) {
    auto node_ts = get_version_number();
    if (node_ts != invalid_ts) { return {true, node_ts}; }
    auto cur_ts = cuda_memory<size_type>::load(camera, cuda_memory_order::memory_order_relaxed);

    size_type old = invalid_ts;
    if (tile_.thread_rank() == version_lane_) {
      old = atomicCAS(
          reinterpret_cast<unsigned int*>(&node_ptr_[version_lane_].first), invalid_ts, cur_ts);
    }
    old          = tile_.shfl(old, version_lane_);
    bool success = old == invalid_ts;
    if (success) {
      set_timestamp(cur_ts);
    } else {
      set_timestamp(old);
    }
    return {success, success ? cur_ts : old};
  }

  DEVICE_QUALIFIER key_type get_low_key() const {
    // highkey is stored as key
    auto low_key_value = get_key_from_lane(0);
    auto low_key       = *reinterpret_cast<key_type*>(&low_key_value);
    return low_key;
  }

  DEVICE_QUALIFIER bool is_lockbit_set(const unsigned_type& data) const {
    return data & lock_bit_mask_;
  }

  DEVICE_QUALIFIER void set_pair(const pair_type pair) { lane_pair_ = pair; }

  DEVICE_QUALIFIER bool is_full() const {
    auto key = get_key_from_lane(last_pair_lane_);
    return key != std::numeric_limits<uint32_t>::max();
  }

  DEVICE_QUALIFIER void set_lock_in_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
      lane_pair_.second = set_lock_bit(lane_pair_.second);
    }
  }
  DEVICE_QUALIFIER void unset_lock_in_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
      lane_pair_.second = unset_lock_bit(lane_pair_.second);
    }
  }

  DEVICE_QUALIFIER void set_leaf_in_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
      lane_pair_.second = set_leaf_bit(lane_pair_.second);
    }
    is_intermediate_ = false;
  }
  DEVICE_QUALIFIER void unset_leaf_in_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
      lane_pair_.second = unset_leaf_bit(lane_pair_.second);
    }
    is_intermediate_ = true;
  }
  DEVICE_QUALIFIER bool is_locked() const {
    bool lock_check = true;
    if (tile_.thread_rank() == metadata_lane_) { lock_check = is_lockbit_set(lane_pair_.second); }
    lock_check = tile_.shfl(lock_check, metadata_lane_);

#ifdef DEBUG_CHECKS
    cuda_assert(is_locked_ == lock_check);
#endif
    return lock_check;
  }

  DEVICE_QUALIFIER bool is_leaf() const { return !is_intermediate_; }

  DEVICE_QUALIFIER bool is_intermediate() const { return is_intermediate_; }

  DEVICE_QUALIFIER void lock() {
    while (auto failed = !try_lock()) {}
    is_locked_ = true;
  }

  // returns true if succeed
  DEVICE_QUALIFIER bool try_lock() {
    unsigned_type old;
    if (tile_.thread_rank() == metadata_lane_) {
      old = atomicOr(reinterpret_cast<unsigned int*>(&node_ptr_[metadata_lane_].second),
                     static_cast<unsigned int>(lock_bit_mask_));
    }
    old        = tile_.shfl(old, metadata_lane_);
    is_locked_ = !is_lockbit_set(old);

    if (is_locked_) {
      set_lock_in_registers();
      __threadfence();
    } else {
      unset_lock_in_registers();
    }

    return is_locked_;
  }
  DEVICE_QUALIFIER void unlock() {
    __threadfence();
    unsigned_type old;
    if (tile_.thread_rank() == metadata_lane_) {
      old = atomicAnd(reinterpret_cast<unsigned int*>(&node_ptr_[metadata_lane_].second),
                      static_cast<unsigned int>(~lock_bit_mask_));
    }
    unset_lock_in_registers();
    is_locked_ = false;

#ifdef DEBUG_CHECKS
    // Node should have been locked
    old             = tile_.shfl(old, metadata_lane_);
    bool was_locked = is_lockbit_set(old);
    cuda_assert(was_locked);
#endif
  }

  DEVICE_QUALIFIER pair_type get_pair() const { return lane_pair_; }

  DEVICE_QUALIFIER btree_versioned_node<pair_type, tile_type, node_width>& operator=(
      const btree_versioned_node<pair_type, tile_type, node_width>& other) {
    node_ptr_        = other.node_ptr_;
    lane_pair_       = other.lane_pair_;
    is_locked_       = other.is_locked_;
    is_intermediate_ = other.is_intermediate_;
    return *this;
  }

  // metadata_lane_ Maps to a pair of {high-key, [lock-bit][leaf-bit][ptr-30bits]}. This
  // has to be the last pair in the node
  static constexpr uint32_t metadata_lane_        = node_width - 1;
  static constexpr uint32_t version_lane_         = node_width - 2;
  static constexpr uint32_t bits_per_byte_        = 8;
  static constexpr uint32_t lock_bit_offset_      = sizeof(key_type) * bits_per_byte_ - 1;
  static constexpr uint32_t leaf_bit_offset_      = lock_bit_offset_ - 1;
  static constexpr uint32_t lock_bit_mask_        = 1u << lock_bit_offset_;  // MSB is lock bit
  static constexpr uint32_t leaf_bit_mask_        = 1u << leaf_bit_offset_;  // 2nd MSB is leaf bit
  static constexpr uint32_t reserved_lanes_count_ = 2;  // one pair reserved for side-link
  static constexpr uint32_t valid_lanes_count_    = node_width - reserved_lanes_count_;
  static constexpr uint32_t last_pair_lane_       = node_width - reserved_lanes_count_ - 1;
  static constexpr int half_node_width_           = node_width >> 1;

 private:
  pair_type* node_ptr_;
  pair_type lane_pair_;
  const tile_type tile_;
  bool is_locked_;
  bool is_intermediate_;
};
