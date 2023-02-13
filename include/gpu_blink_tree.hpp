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
#define _CG_ABI_EXPERIMENTAL  // enable experimental CGs API

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <btree_kernels.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <node.hpp>
#include <pair_type.hpp>
#include <queue>
#include <sstream>
#include <type_traits>

#include <device_bump_allocator.hpp>
#include <slab_alloc.hpp>

// #define DEBUG_LOCKS
//  #define DEBUG_STRUCTURE

#ifdef DEBUG_STRUCTURE
#define DEBUG_STRUCTURE_PRINT(fmt, ...)         \
  if (tile.thread_rank() == 0) {                \
    do { printf(fmt, __VA_ARGS__); } while (0); \
  }
#else
#define DEBUG_STRUCTURE_PRINT(fmt, ...)
#endif

#define DEBUG_THREAD_TO_TILE ((threadIdx.x + blockIdx.x * blockDim.x) / branching_factor)

#ifdef DEBUG_LOCKS
#define debug_print_tile(tile, fmt, ...)        \
  if (tile.thread_rank() == 0) {                \
    do { printf(fmt, __VA_ARGS__); } while (0); \
  }
#else
#define debug_print_tile(fmt, ...)
#endif

namespace GpuBTree {

template <typename Key,
          typename Value,
          int B              = 16,
          typename Allocator = device_bump_allocator<node_type<Key, Value, B>>>
struct gpu_blink_tree {
  using size_type                        = uint32_t;
  using key_type                         = Key;
  using value_type                       = Value;
  using pair_type                        = pair_type<Key, Value>;
  static auto constexpr branching_factor = B;

  static constexpr key_type invalid_key     = std::numeric_limits<key_type>::max();
  static constexpr value_type invalid_value = std::numeric_limits<key_type>::max();

  using allocator_type                = Allocator;
  using device_allocator_context_type = device_allocator_context<allocator_type>;
  gpu_blink_tree() : allocator_{} {
    static_assert(sizeof(Key) == sizeof(Value),
                  "Size of key must be the same as the size of the value");
    allocate();
  }
  // Bulk build is only supported for device_bump_allocator
  gpu_blink_tree(const Key* keys,
                 const Value* values,
                 const size_type num_keys,
                 const bool sorted_input = false,
                 cudaStream_t stream     = 0)
      : allocator_{} {
    static_assert(sizeof(Key) == sizeof(Value),
                  "Size of key must be the same as the size of the value");
    allocate();

    // Build a half-full tree
    static constexpr uint32_t bulk_build_branching_factor     = 8;
    static constexpr uint32_t log_bulk_build_branching_factor = 3;

    size_type num_keys_and_zero = num_keys + 1;  // tree must include the minimum possible key

    uint32_t num_leaves = (num_keys_and_zero % bulk_build_branching_factor)
                              ? num_keys_and_zero / bulk_build_branching_factor + 1
                              : num_keys_and_zero / bulk_build_branching_factor;
    uint32_t num_nodes          = num_leaves;
    uint32_t num_interior_nodes = num_leaves;

    uint32_t tree_height = 0;
    for (tree_height = 0; num_interior_nodes != 1; tree_height++) {
      int frac = (num_interior_nodes % bulk_build_branching_factor) ? 1 : 0;
      num_interior_nodes >>= log_bulk_build_branching_factor;
      num_interior_nodes += frac;
      num_nodes += num_interior_nodes;
      if (num_interior_nodes == 1) break;
    }
    tree_height += 2;

    static_assert(branching_factor == 16);

    const uint32_t block_size = 256;
    const uint32_t num_blocks = (num_nodes * 32 + block_size - 1) / block_size;

    if (!sorted_input) {
      std::cout << "Only sorted input is supported" << std::endl;
      std::terminate();
    }
    kernels::bulk_build_kernel<<<num_blocks, block_size, 0, stream>>>(
        keys,
        values,
        num_keys_and_zero,
        num_nodes,
        num_leaves,
        tree_height,
        bulk_build_branching_factor,
        log_bulk_build_branching_factor,
        *this);
  }

  gpu_blink_tree(const gpu_blink_tree& other)
      : root_index_(other.root_index_)
      , h_btree_(other.h_btree_)
      , h_node_count_(other.h_node_count_)
      , d_root_index_(other.d_root_index_)
      , allocator_(other.allocator_) {}

  ~gpu_blink_tree() {}

  // host-side APIs
  void concurrent_insert_range(const Key* keys,
                               const Value* values,
                               const size_type num_insertion,
                               const Key* lower_bound,
                               const Key* upper_bound,
                               const size_type num_ranges,
                               pair_type* result,
                               const size_type average_range_length,
                               cudaStream_t stream = 0) {
    int block_size = 128;

    // num_blocks is hardware specific
    int num_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        kernels::concurrent_insert_range_kernel_blink<
            Key,
            Value,
            pair_type,
            size_type,
            typename std::remove_reference<decltype(*this)>::type>,
        block_size,
        0);
    int device_id = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    auto sms_count            = device_prop.multiProcessorCount;
    const uint32_t num_blocks = num_blocks_per_sm * sms_count;

    kernels::concurrent_insert_range_kernel_blink<<<num_blocks, block_size, 0, stream>>>(
        keys,
        values,
        num_insertion,
        lower_bound,
        upper_bound,
        num_ranges,
        result,
        average_range_length,
        *this);
  }

  void concurrent_find_erase(const Key* find_keys,
                             Value* find_results,
                             const size_type num_finds,
                             const Key* erase_keys,
                             const size_type num_erasures,
                             cudaStream_t stream = 0) {
    int block_size = 128;

    // num_blocks is hardware specific
    int num_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        kernels::concurrent_find_erase_kernel_blink<
            Key,
            Value,
            size_type,
            typename std::remove_reference<decltype(*this)>::type>,
        block_size,
        0);
    int device_id = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    auto sms_count            = device_prop.multiProcessorCount;
    const uint32_t num_blocks = num_blocks_per_sm * sms_count;

    kernels::concurrent_find_erase_kernel_blink<<<num_blocks, block_size, 0, stream>>>(
        find_keys, find_results, num_finds, erase_keys, num_erasures, *this);
  }

  void insert(const Key* keys,
              const Value* values,
              const size_type num_keys,
              cudaStream_t stream = 0) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::insert_kernel<<<num_blocks, block_size, 0, stream>>>(keys, values, num_keys, *this);
  }

  void find(const Key* keys,
            Value* values,
            const size_type num_keys,
            cudaStream_t stream = 0,
            bool concurrent     = false) const {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::find_kernel<<<num_blocks, block_size, 0, stream>>>(
        keys, values, num_keys, *this, concurrent);
  }

  void erase(const Key* keys,
             const size_type num_keys,
             cudaStream_t stream = 0,
             bool concurrent     = false) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::erase_kernel<<<num_blocks, block_size, 0, stream>>>(keys, num_keys, *this, concurrent);
  }
  // [lower_bound, upper_bound)
  void range_query(const Key* lower_bound,
                   const Key* upper_bound,
                   pair_type* result,
                   size_type* counts,
                   const size_type average_range_length,
                   const size_type num_keys,
                   cudaStream_t stream = 0,
                   bool concurrent     = false) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::range_query_kernel<<<num_blocks, block_size, 0, stream>>>(lower_bound,
                                                                       upper_bound,
                                                                       result,
                                                                       average_range_length,
                                                                       num_keys,
                                                                       *this,
                                                                       counts,
                                                                       concurrent);
  }

  // Device-side APIs
  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool cooperative_insert(const Key& key,
                                           const Value& value,
                                           const tile_type& tile,
                                           DeviceAllocator& allocator) {
    return cooperative_insert_blink(key, value, tile, allocator);
    // return cooperative_insert_hand_over_hand(key, value, tile, allocator);
  }

  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER Value cooperative_find(const Key& key,
                                          const tile_type& tile,
                                          DeviceAllocator& allocator,
                                          bool concurrent = false) {
    auto value              = std::numeric_limits<uint32_t>::max();
    using node_type         = btree_node<pair_type, tile_type, branching_factor>;
    auto current_node_index = *d_root_index_;
    while (true) {
      node_type current_node = node_type(
          reinterpret_cast<pair_type*>(allocator.address(allocator_, current_node_index)), tile);
      if (concurrent) {
        current_node.load(cuda_memory_order::memory_order_relaxed);
        traverse_side_links(current_node, current_node_index, key, tile, allocator);
      } else {
        current_node.load();
      }
      bool is_leaf = current_node.is_leaf();
      if (is_leaf) {
        value = current_node.get_key_value_from_node(key);
        return value;
      } else {
        current_node_index = current_node.find_next(key);
      }
    }
    return value;
  }

  // Range query that includes [lower_bound, upper_bound)
  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER size_type cooperative_range_query(const Key& lower_bound,
                                                     const Key& upper_bound,
                                                     const tile_type& tile,
                                                     DeviceAllocator& allocator,
                                                     pair_type* buffer = nullptr,
                                                     bool concurrent   = false) /*const*/
  {
    using node_type         = btree_node<pair_type, tile_type, branching_factor>;
    auto current_node_index = *d_root_index_;
    size_type count         = 0;
    while (true) {
      node_type current_node = node_type(
          reinterpret_cast<pair_type*>(allocator.address(allocator_, current_node_index)), tile);
      if (concurrent) {
        current_node.load(cuda_memory_order::memory_order_relaxed);
        traverse_side_links(current_node, current_node_index, lower_bound, tile, allocator);
      } else {
        current_node.load();
      }
      bool is_leaf = current_node.is_leaf();
      if (is_leaf) {
        bool keep_traversing = true;
        do {
          if (buffer != nullptr) {
            count += current_node.get_in_range(lower_bound, upper_bound, buffer + count);
          } else {
            count += current_node.get_in_range(lower_bound, upper_bound, nullptr);
          }
          keep_traversing = upper_bound > current_node.get_high_key();
          if (keep_traversing) {
            current_node_index = current_node.get_sibling_index();
            current_node       = node_type(
                reinterpret_cast<pair_type*>(allocator.address(allocator_, current_node_index)),
                tile);
            if (concurrent) {
              current_node.load(cuda_memory_order::memory_order_relaxed);
            } else {
              current_node.load();
            }
          }
        } while (keep_traversing);

        return count;
      } else {
        current_node_index = current_node.find_next(lower_bound);
      }
    }
    return count;
  }

  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool cooperative_erase(const Key& key,
                                          const tile_type& tile,
                                          DeviceAllocator& allocator,
                                          bool concurrent = false) {
    using node_type         = btree_node<pair_type, tile_type, branching_factor>;
    auto current_node_index = *d_root_index_;
    while (true) {
      node_type current_node = node_type(
          reinterpret_cast<pair_type*>(allocator.address(allocator_, current_node_index)), tile);
      if (concurrent) {
        current_node.load(cuda_memory_order::memory_order_relaxed);
        traverse_side_links(current_node, current_node_index, key, tile, allocator);
      } else {
        current_node.load();
      }
      bool is_leaf   = current_node.is_leaf();
      bool is_locked = current_node.is_locked();
      if (is_leaf) {
        current_node.lock();
        current_node.load(cuda_memory_order::memory_order_relaxed);
        if (concurrent) {
          traverse_side_links_with_locks(current_node, current_node_index, key, tile, allocator);
        }
        bool success = current_node.erase(key);
        if (success) { current_node.store(cuda_memory_order::memory_order_relaxed); }
        current_node.unlock();
        return success;
      } else {
        current_node_index = current_node.find_next(key);
      }
    }
    return false;
  }

  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool cooperative_insert_hand_over_hand(const Key& key,
                                                          const Value& value,
                                                          const tile_type& tile,
                                                          DeviceAllocator& allocator) {
    using node_type         = btree_node<pair_type, tile_type, branching_factor>;
    auto root_index         = *d_root_index_;
    auto current_node_index = root_index;
    auto parent_index       = root_index;
    bool keep_going         = true;
    do {
      auto current_node = node_type(
          reinterpret_cast<pair_type*>(allocator.address(allocator_, current_node_index)), tile);
      current_node.lock();
      current_node.load(cuda_memory_order::memory_order_relaxed);
      bool is_intermediate = current_node.is_intermediate();
      bool is_full         = current_node.is_full();
      if (is_full) {
        if (current_node_index == root_index) {
          auto sibling_index0 = allocator.allocate(allocator_, 1, tile);
          auto sibling_index1 = allocator.allocate(allocator_, 1, tile);
          auto two_siblings =
              current_node.split_as_root(sibling_index0,  // left node
                                         sibling_index1,  // left right
                                         reinterpret_cast<pair_type*>(allocator.address(
                                             allocator_, sibling_index0)),  // left ptr
                                         reinterpret_cast<pair_type*>(allocator.address(
                                             allocator_, sibling_index1)),  // right ptr
                                         true);                             // children_are_locked

          two_siblings.right.store(cuda_memory_order::memory_order_relaxed);
          two_siblings.left.store(cuda_memory_order::memory_order_relaxed);
          current_node.store(cuda_memory_order::memory_order_relaxed);  // root is still locked
          current_node.unlock();

          // go right or left?
          current_node_index = current_node.find_next(key);
          if (current_node_index == sibling_index0) {  // go left and keep root locked
            two_siblings.right.unlock();
            current_node = two_siblings.left;
          } else {  // go right and keep root locked
            two_siblings.left.unlock();
            current_node = two_siblings.right;
          }
        } else {
          auto sibling_index = allocator.allocate(allocator_, 1, tile);
          auto go_right      = current_node.key_is_in_upperhalf(key);
          auto split_result  = current_node.split(
              sibling_index,
              parent_index,
              reinterpret_cast<pair_type*>(allocator.address(allocator_, sibling_index)),
              reinterpret_cast<pair_type*>(allocator.address(allocator_, parent_index)),
              true);

          split_result.sibling.store(cuda_memory_order::memory_order_relaxed);
          current_node.store(cuda_memory_order::memory_order_relaxed);
          split_result.parent.store(cuda_memory_order::memory_order_relaxed);
          split_result.parent.unlock();

          if (go_right) {
            current_node_index = sibling_index;
            current_node.unlock();
            current_node = split_result.sibling;
          } else {
            split_result.sibling.unlock();
          }
        }
      } else {
        // it is safe to release the parent's lock now
        if (parent_index != current_node_index) {
          auto parent_node = node_type(
              reinterpret_cast<pair_type*>(allocator.address(allocator_, parent_index)), tile);
          parent_node.unlock();
        }
      }
      if (is_intermediate) {  // traverse
        parent_index       = current_node_index;
        current_node_index = current_node.find_next(key);
      } else {
        current_node.insert(key, value);
        current_node.store(cuda_memory_order::memory_order_relaxed);
        current_node.unlock();

        return true;
      }
    } while (keep_going);
    return false;
  }

  // Tries to traverse the side-links without locks
  // Return true if a side-link was traversed
  template <typename tile_type, typename node_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool traverse_side_links(node_type& node,
                                            size_type& node_index,
                                            const Key& key,
                                            const tile_type& tile,
                                            DeviceAllocator& allocator) {
    bool traversed = false;
    while (key >= node.get_high_key()) {
      node_index = node.get_sibling_index();
      node =
          node_type(reinterpret_cast<pair_type*>(allocator.address(allocator_, node_index)), tile);
      node.load(cuda_memory_order::memory_order_relaxed);
      traversed |= true;
    }
    return traversed;
  }

  // Tries to traverse the side-links with locks
  // Return true if a side-link was traversed
  template <typename tile_type, typename node_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool traverse_side_links_with_locks(node_type& node,
                                                       size_type& node_index,
                                                       const Key& key,
                                                       const tile_type& tile,
                                                       DeviceAllocator& allocator) {
    bool traversed = false;
    while (key >= node.get_high_key()) {
      node_index = node.get_sibling_index();
      node_type sibling_node =
          node_type(reinterpret_cast<pair_type*>(allocator.address(allocator_, node_index)), tile);
      sibling_node.lock();
      node.unlock();
      node = sibling_node;
      node.load(cuda_memory_order::memory_order_relaxed);
      traversed |= true;
    }
    return traversed;
  }

  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool cooperative_insert_blink(const Key& key,
                                                 const Value& value,
                                                 const tile_type& tile,
                                                 DeviceAllocator& allocator) {
    using node_type         = btree_node<pair_type, tile_type, branching_factor>;
    auto root_index         = *d_root_index_;
    auto current_node_index = root_index;
    auto parent_index       = root_index;
    bool keep_going         = true;
    bool link_traversed     = false;
    do {
      auto current_node = node_type(
          reinterpret_cast<pair_type*>(allocator.address(allocator_, current_node_index)), tile);
      current_node.load(cuda_memory_order::memory_order_relaxed);

      // if we restarted from root, we reset the traversal
      link_traversed = current_node_index == root_index ? false : link_traversed;

      // Traversing side-links
      link_traversed |= traverse_side_links(current_node, current_node_index, key, tile, allocator);

      bool is_leaf = current_node.is_leaf();
      if (is_leaf) {
        if (current_node.try_lock()) {
          current_node.load(cuda_memory_order::memory_order_relaxed);
          bool parent_unknown =
              current_node_index == parent_index && current_node_index != root_index;
          bool traversal_required = key >= current_node.get_high_key();
          // if the parent is unknown we will not proceed
          if (parent_unknown && traversal_required) {
            current_node.unlock();
            current_node_index = root_index;
            parent_index       = root_index;
            continue;
          }
          is_leaf = current_node.is_leaf();
          // if the node is not a leaf anymore, we don't need the lock
          if (!is_leaf) { current_node.unlock(); }

          // traversal while holding the lock
          while (key >= current_node.get_high_key()) {
            if (is_leaf) { current_node.unlock(); }
            current_node_index = current_node.get_sibling_index();
            current_node       = node_type(
                reinterpret_cast<pair_type*>(allocator.address(allocator_, current_node_index)),
                tile);
            if (is_leaf) { current_node.lock(); }
            current_node.load(cuda_memory_order::memory_order_relaxed);
            is_leaf = current_node.is_leaf();
            // if the node is not a leaf anymore, we don't need the lock
            if (!is_leaf) { current_node.unlock(); }
            link_traversed = true;
          }
        } else {
          current_node_index = parent_index;
          continue;
        }
      }

      // make sure that if the node is full, we know the parent
      // we only know the parent if we didn't do side-traversal
      bool is_full = current_node.is_full();
      if (is_full && link_traversed) {
        if (is_leaf) {
          current_node.unlock();
          current_node_index = root_index;
          parent_index       = root_index;
          continue;
        }
      }

      // if is full, and not leaf we need to acquire the lock
      if (is_full && !is_leaf) {
        if (current_node.try_lock()) {
          current_node.load(cuda_memory_order::memory_order_relaxed);
          is_full = current_node.is_full();
          if (is_full) {
            bool traversal_required = key >= current_node.get_high_key();
            // if we traverse, parent will change so we will restart
            if (traversal_required) {
              current_node.unlock();
              current_node_index = root_index;
              parent_index       = root_index;
              continue;
            }
          } else {
            current_node.unlock();
            // Traversing side-links
            link_traversed |=
                traverse_side_links(current_node, current_node_index, key, tile, allocator);
          }
        } else {
          current_node_index = parent_index;
          continue;
        }
      }

      is_full = current_node.is_full();
      // if the node full after we restarted we can't proceed
      if (is_full && (current_node_index != root_index) && (current_node_index == parent_index)) {
        current_node.unlock();
        current_node_index = root_index;
        parent_index       = root_index;
        continue;
      }

      // splitting an intermediate node
      if (is_full && (current_node_index != root_index)) {
        auto parent_node = node_type(
            reinterpret_cast<pair_type*>(allocator.address(allocator_, parent_index)), tile);
        parent_node.lock();
        parent_node.load(cuda_memory_order::memory_order_relaxed);
        bool parent_is_full = parent_node.is_full();

        // make sure parent is not full
        if (parent_is_full) {
          current_node.unlock();
          parent_node.unlock();
          current_node_index = root_index;
          parent_index       = root_index;
          continue;
        }

        // make sure parent is correct parent
        auto parent_is_correct = parent_node.ptr_is_in_node(current_node_index);
        if (!parent_is_correct) {
          current_node.unlock();
          parent_node.unlock();
          current_node_index = root_index;
          parent_index       = root_index;
          continue;
        }

        // now it is safe to split
        auto sibling_index = allocator.allocate(allocator_, 1, tile);
        auto go_right      = current_node.key_is_in_upperhalf(key);

        auto split_result = current_node.split(
            sibling_index,
            parent_index,
            reinterpret_cast<pair_type*>(allocator.address(allocator_, sibling_index)),
            reinterpret_cast<pair_type*>(allocator.address(allocator_, parent_index)),
            true);

        split_result.sibling.store(cuda_memory_order::memory_order_relaxed);
        __threadfence();
        current_node.store(cuda_memory_order::memory_order_relaxed);
        __threadfence();
        split_result.parent.store(cuda_memory_order::memory_order_relaxed);
        split_result.parent.unlock();

        if (go_right) {
          current_node_index = sibling_index;
          current_node.unlock();
          current_node = split_result.sibling;
        } else {
          split_result.sibling.unlock();
        }

        is_leaf = current_node.is_leaf();
        if (!is_leaf) { current_node.unlock(); }
      } else if (is_full) {
        auto sibling_index0 = allocator.allocate(allocator_, 1, tile);
        auto sibling_index1 = allocator.allocate(allocator_, 1, tile);
        auto two_siblings =
            current_node.split_as_root(sibling_index0,  // left node
                                       sibling_index1,  // left right
                                       reinterpret_cast<pair_type*>(allocator.address(
                                           allocator_, sibling_index0)),  // left ptr
                                       reinterpret_cast<pair_type*>(allocator.address(
                                           allocator_, sibling_index1)),  // right ptr
                                       true);                             // children_are_locked

        // write order here should be:
        // right node -> left node ->  parent
        two_siblings.right.store(cuda_memory_order::memory_order_relaxed);
        __threadfence();
        two_siblings.left.store(cuda_memory_order::memory_order_relaxed);
        __threadfence();
        current_node.store(cuda_memory_order::memory_order_relaxed);  // root is still locked
        current_node.unlock();

        // go right or left?
        current_node_index = current_node.find_next(key);
        if (current_node_index == sibling_index0) {  // go left
          two_siblings.right.unlock();
          current_node = two_siblings.left;
        } else {  // go right
          two_siblings.left.unlock();
          current_node = two_siblings.right;
        }
        parent_index = root_index;
        is_leaf      = current_node.is_leaf();
        if (!is_leaf) { current_node.unlock(); }
      }

      // traversal and insertion
      is_leaf = current_node.is_leaf();
      if (is_leaf) {
        current_node.insert(key, value);
        current_node.store(cuda_memory_order::memory_order_relaxed);
        current_node.unlock();
        return true;
      } else {  // traverse
        parent_index       = link_traversed ? root_index : current_node_index;
        current_node_index = current_node.find_next(key);
      }
    } while (keep_going);

    return false;
  }

  // Stats
  void copy_tree_to_host(size_type bytes_count) {
    allocator_.copy_buffer(reinterpret_cast<node_type<Key, Value, B>*>(h_btree_), bytes_count);
  }

  size_type get_num_tree_node() { return allocator_.get_allocated_count(); }

  template <typename Func>
  bool validate_tree_structure(const std::vector<key_type>& keys,
                               Func to_value,
                               bool quiet = false) {
    auto num_allocated_nodes = get_num_tree_node();
    // slab alloc currently doesn't have this implemented
    if (num_allocated_nodes == 0) { return false; }
    h_btree_ = new pair_type[num_allocated_nodes * branching_factor];

    std::size_t tree_size = num_allocated_nodes * branching_factor;
    tree_size *= (sizeof(Key) + sizeof(Value));

    copy_tree_to_host(tree_size);

    std::cout << "Copying " << num_allocated_nodes << " nodes" << std::endl;
    std::cout << "        " << double(tree_size) / double(1 << 30) << " GBs" << std::endl;

    size_type num_nodes       = 0;
    size_type prev_level      = branching_factor - 1;
    size_type pairs_count     = 0;
    size_type pairs_per_level = branching_factor;
    size_type cur_level       = 0;
    size_type level_nodes     = 0;
    size_type added_to_level  = 0;
    size_type tree_height     = 0;

    std::vector<size_type> level_nodes_ids;
    std::vector<std::vector<size_type>> levels_nodes_ids;

    std::vector<pair_type> tree_pairs;
    std::queue<size_type> queue;

    // Copied from node struct because we can't construct it on CPU
    uint32_t bits_per_byte   = 8;
    uint32_t lock_bit_offset = sizeof(key_type) * bits_per_byte - 1;
    uint32_t leaf_bit_offset = lock_bit_offset - 1;
    uint32_t lock_bit_mask   = 1u << lock_bit_offset;
    uint32_t leaf_bit_mask   = 1u << leaf_bit_offset;
    uint32_t metadata_lane   = branching_factor - 1;

    auto is_locked_node = [&](pair_type pair) { return pair.second & lock_bit_mask; };
    auto is_leaf_node   = [leaf_bit_mask](pair_type pair) { return pair.second & leaf_bit_mask; };
    auto mask_metadata  = [leaf_bit_mask, lock_bit_mask](pair_type pair) {
      auto mask = (~lock_bit_mask) & (~leaf_bit_mask);
      return pair_type(pair.first, pair.second & mask);
    };
    auto tombstone_pair      = pair_type{0, 0};
    auto invalid_pair        = pair_type{};
    auto empty_metadata_pair = mask_metadata(invalid_pair);

    queue.push(0);

    while (!queue.empty()) {
      pair_type* current = h_btree_ + queue.front() * branching_factor;
      num_nodes++;
      level_nodes_ids.push_back(queue.front());
      queue.pop();
      level_nodes++;

      pair_type metadata = *(current + metadata_lane);
      bool is_leaf       = is_leaf_node(metadata);
      bool is_locked     = is_locked_node(metadata);

      if (is_locked) { throw std::logic_error("Tree node is locked"); }

      if (current->first == tombstone_pair.first) { tree_height++; }

      for (uint32_t pair_id = 0; pair_id < metadata_lane; pair_id++) {
        if (*current != invalid_pair) {
          bool is_tombstone = (*current) == tombstone_pair;
          if (is_leaf) {
            if (!is_tombstone) { tree_pairs.push_back(*current); }
            if (tree_pairs.size() == 0 && !is_tombstone) {
              throw std::logic_error("First pair is not the tombstone pair");
            }
          } else {
            queue.push(current->second);
            added_to_level++;
          }
        }
        pairs_count++;
        current++;
      }
      if (pairs_count == prev_level) {
        pairs_count = 0;
        pairs_per_level *= (branching_factor - 1);
        cur_level++;
        levels_nodes_ids.push_back(level_nodes_ids);
        level_nodes_ids.clear();
        level_nodes    = 0;
        prev_level     = added_to_level * (branching_factor - 1);
        added_to_level = 0;
      }
    }

    if (tree_pairs.size() != keys.size()) {
      throw std::logic_error("Number of keys in the tree is not the same as the input");
    }

    // Validate the tree structure
    for (uint32_t level_id = 0; level_id < levels_nodes_ids.size(); level_id++) {
      for (uint32_t node_id = 0; node_id < levels_nodes_ids[level_id].size(); node_id++) {
        uint32_t node_idx  = levels_nodes_ids[level_id][node_id];
        pair_type metadata = (h_btree_ + node_idx * branching_factor)[metadata_lane];
        metadata           = mask_metadata(metadata);

        key_type link_min  = metadata.first;
        size_type link_ptr = metadata.second;

        if (node_id == (levels_nodes_ids[level_id].size() - 1)) {  // last node in level
          if (metadata != empty_metadata_pair) {
            throw std::logic_error("Invalid link information at the end of the level");
          }
        } else {
          size_type correct_ptr = levels_nodes_ids[level_id][node_id + 1];
          if (link_ptr != correct_ptr) {
            throw std::logic_error("Invalid link information at node");
          }
          pair_type* neighbor_node = (h_btree_ + correct_ptr * branching_factor);
          pair_type* cur_node      = (h_btree_ + node_idx * branching_factor);
          for (uint32_t i = 0; i < branching_factor; i++) {
            auto cur_pair = (i == (branching_factor - 1)) ? mask_metadata(*cur_node) : *cur_node;
            auto neighbor_pair =
                (i == (branching_factor - 1)) ? mask_metadata(*neighbor_node) : *neighbor_node;

            if (neighbor_pair != invalid_pair && neighbor_pair.first < link_min) {
              throw std::logic_error("Invalid link information at node");
            }
            if (i < (branching_factor - 1) && cur_pair != invalid_pair &&
                cur_pair.first >= link_min) {
              throw std::logic_error("Invalid link information at node");
            }
            neighbor_node++;
            cur_node++;
          }
        }
      }
    }

    auto sorted_keys = keys;
    std::sort(sorted_keys.begin(), sorted_keys.end());

    for (uint32_t key_id = 0; key_id < keys.size(); key_id++) {
      auto key           = sorted_keys[key_id];
      auto expected_pair = pair_type{key, to_value(key)};
      auto found_pair    = tree_pairs[key_id];
      if (expected_pair != found_pair) { throw std::logic_error("Input pair mismatch tree pair"); }
    }
    if (!quiet) {
      std::cout << "B-Tree height = " << tree_height << std::endl;
      for (uint32_t level_id = 0; level_id < levels_nodes_ids.size(); level_id++) {
        std::cout << "Level: " << level_id << " -> " << levels_nodes_ids[level_id].size()
                  << " nodes" << std::endl;
      }
      std::cout << "B-Tree structure is correct" << std::endl;
    }
    delete[] h_btree_;
    return true;
  }

  void plot_node(std::stringstream& dot,
                 pair_type* node,
                 size_type id,
                 const bool plot_links,
                 const bool plot_box_only = false) {
    uint32_t locked_bit_mask = (1u << 31);
    uint32_t leaf_bit_mask   = (1u << 30);
    uint32_t empty_pointer   = ((~locked_bit_mask) & (~leaf_bit_mask));
    std::vector<key_type> connectivity;
    dot << "node" << id << " [shape=none "
        << "xlabel=\"" << id << "\" label=<<";
    dot << "table border = \"0\" cellspacing=\"0\"><tr>";

    for (size_t lane = 0; lane < branching_factor; lane++) {
      bool ptr_lane    = lane == (branching_factor - 1);
      bool ts_lane     = false;
      key_type key     = node[lane].first;
      value_type value = node[lane].second;

      bool is_locked = node[branching_factor - 1].second & locked_bit_mask;
      bool is_leaf   = node[branching_factor - 1].second & leaf_bit_mask;
      auto ptr       = value & empty_pointer;

      //<tr><td port = "port1" border = "1" bgcolor = "red"> corpus_language</ td></ tr>
      dot << std::endl;
      if (!plot_box_only) {
        dot << "<td port = \"f" << lane << "\" ";
        dot << "border=\"1\"";
        if (ptr_lane) {
          dot << " bgcolor=\"#00BFFF	\"";
        } else if (ts_lane) {
          dot << " bgcolor=\"#00FFFF\"";
        } else if (is_locked) {
          dot << " bgcolor=\"red\"";
        } else if (is_leaf) {
          dot << " bgcolor=\"darkolivegreen1\"";
        } else {
          dot << " bgcolor=\"#D3D3D3\"";
        }
        dot << ">";
        if (key == invalid_key) {
          dot << " ";
        } else {
          if (ts_lane) {
            dot << " ts=" << key;
          } else {
            if (ptr_lane && !plot_links) {
              dot << " " << key << "," << ptr;
            } else {
              dot << " " << key;
            }
          }
        }
        dot << "</td>";
      }

      if (is_leaf && !ptr_lane && !ts_lane) {
        connectivity.push_back(empty_pointer);
      } else {
        connectivity.push_back(ptr);
      }
    }
    dot << std::endl;
    dot << "</tr></table>>]" << std::endl;
    // connectivity
    size_t ptrs_to_plot = branching_factor;
    if (!plot_links) ptrs_to_plot--;
    for (size_t lane = 0; lane < ptrs_to_plot; lane++) {
      auto ptr         = connectivity[lane];
      auto is_valid    = ptr != empty_pointer;
      bool ptr_lane    = lane == (branching_factor - 1);
      bool ts_lane     = false;
      auto target_lane = ptr_lane ? 0 : (branching_factor >> 1);
      target_lane      = ts_lane ? branching_factor - 2 : target_lane;
      if (is_valid) {
        dot << "node" << id << ":f" << lane;
        dot << "->";
        dot << "node" << ptr << ":f" << target_lane;
        if (ts_lane) { dot << "[color = \"darkred\"]"; }
        dot << std::endl;
      }
    }
  }

  void plot_dot(std::string fname, const bool plot_links = true) {
    auto num_nodes = get_num_tree_node();
    h_btree_       = new pair_type[num_nodes * branching_factor];
    copy_tree_to_host(num_nodes * branching_factor * (sizeof(Key) + sizeof(Value)));

    std::stringstream dot;
    dot << "digraph g {" << std::endl;
    dot << "forcelabels=true" << std::endl;
    dot << "\t node [height=.05 shape=record]" << std::endl;
    for (size_type node = 0; node < num_nodes; node++) {
      auto node_ts = h_btree_[node * branching_factor + branching_factor - 2].first;
      plot_node(dot, &h_btree_[node * branching_factor], node, plot_links);
    }
    dot << "}" << std::endl;

    std::fstream output(fname + ".gv", std::ios::out);
    output << dot.rdbuf();
    output.close();

    std::string command = "dot ";
    command += fname + ".gv ";
    command += "-Tpdf  -o ";
    command += fname + ".pdf";
    auto res = system(command.c_str());
    (void)res;
    delete[] h_btree_;
  }

  void print_tree_nodes() {
    // allocate

    auto num_nodes = get_num_tree_node();
    h_btree_       = new pair_type[num_nodes * branching_factor];

    copy_tree_to_host(num_nodes * branching_factor * (sizeof(Key) + sizeof(Value)));
    for (size_type node = 0; node < num_nodes; node++) {
      std::cout << "Node: " << node << std::endl;
      for (size_type lane = 0; lane < branching_factor; lane++) { std::cout << lane << ","; }
      std::cout << std::endl;
      for (size_type lane = 0; lane < branching_factor; lane++) {
        bool ptr_lane = lane == (branching_factor - 1);

        auto key   = h_btree_[node * branching_factor + lane].first;
        auto value = h_btree_[node * branching_factor + lane].second;

        uint32_t locked_bit_mask = (1u << 31);
        uint32_t leaf_bit_mask   = (1u << 30);
        uint32_t empty_pointer   = ((~locked_bit_mask) & (~leaf_bit_mask));
        bool is_locked           = value & locked_bit_mask;
        bool is_leaf             = value & leaf_bit_mask;
        value                    = value & empty_pointer;

        if (ptr_lane) {
          if (key == invalid_key) {
            std::cout << "{x,";
          } else {
            std::cout << "{" << key << ",";
          }
          if (value == empty_pointer) {
            std::cout << "x";
          } else {
            std::cout << value;
          }
          if (is_leaf) {
            std::cout << "} -> leaf";
          } else {
            std::cout << "} -> intermediate";
          }
          if (is_locked) {
            std::cout << " -> locked" << std::endl;
          } else {
            std::cout << " -> unlocked" << std::endl;
          }

        } else {
          if (key == invalid_key) {
            std::cout << "{x,x}";
          } else {
            std::cout << "{" << key << "," << value << "}";
          }
        }

        std::cout << std::dec;
      }
      std::cout << "-------------------" << std::endl;
    }
    std::cout << "*********************************************" << std::endl;

    delete[] h_btree_;
  }
  double compute_memory_usage() {
    auto num_nodes       = get_num_tree_node();
    double tree_size_gbs = double(num_nodes) * sizeof(node_type<Key, Value, B>) / (1ull << 30);
    return tree_size_gbs;
  }

 private:
  template <typename key_type,
            typename value_type,
            typename pair_type,
            typename size_type,
            typename btree>
  friend __global__ void kernels::concurrent_insert_range_kernel_blink(const key_type*,
                                                                       const value_type*,
                                                                       const size_type,
                                                                       const key_type*,
                                                                       const key_type*,
                                                                       const size_type,
                                                                       pair_type*,
                                                                       const size_type,
                                                                       btree);

  template <typename key_type, typename value_type, typename size_type, typename btree>
  friend __global__ void kernels::concurrent_find_erase_kernel_blink(const key_type*,
                                                                     value_type*,
                                                                     const size_type,
                                                                     const key_type*,
                                                                     const size_type,
                                                                     btree);

  template <typename key_type, typename value_type, typename size_type, typename btree>
  friend __global__ void kernels::insert_kernel(const key_type*,
                                                const value_type*,
                                                const size_type,
                                                btree);

  template <typename key_type, typename value_type, typename btree>
  friend __global__ void kernels::bulk_build_kernel(const key_type*,
                                                    const value_type*,
                                                    const uint32_t,
                                                    const uint32_t,
                                                    const uint32_t,
                                                    const uint32_t,
                                                    const uint32_t,
                                                    const uint32_t,
                                                    btree);
  template <typename btree>
  friend __global__ void kernels::initialize_kernel(btree);

  template <typename key_type, typename value_type, typename size_type, typename btree>
  friend __global__ void kernels::find_kernel(const key_type*,
                                              value_type*,
                                              const size_type,
                                              btree,
                                              const size_type,
                                              const bool);
  template <typename key_type, typename value_type, typename size_type, typename btree>
  friend __global__ void kernels::find_kernel(const key_type*,
                                              value_type*,
                                              const size_type,
                                              const btree,
                                              const bool);
  template <typename key_type, typename pair_type, typename size_type, typename btree>
  friend __global__ void kernels::range_query_kernel(const key_type*,
                                                     const key_type*,
                                                     pair_type*,
                                                     const size_type,
                                                     const size_type,
                                                     btree,
                                                     size_type*,
                                                     const bool);

  template <typename key_type, typename size_type, typename btree>
  friend __global__ void kernels::erase_kernel(const key_type*, const size_type, btree, const bool);

  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER void allocate_root_node(const tile_type& tile, DeviceAllocator& allocator) {
    auto root_index = allocator.allocate(allocator_, 1, tile);
    *d_root_index_  = root_index;
    using node_type = btree_node<pair_type, tile_type, branching_factor>;

    auto lane_pair = pair_type();
    if (tile.thread_rank() == 0) { lane_pair = pair_type{0, 0}; }
    auto root_node =
        node_type(reinterpret_cast<pair_type*>(allocator.address(allocator_, root_index)),
                  tile,
                  lane_pair,
                  false,
                  false);
    root_node.unset_lock_in_registers();
    root_node.set_leaf_in_registers();
    root_node.store();
  }

  void allocate() {
    d_root_index_ = cuda_allocator<size_type>().allocate(1);
    cuda_try(cudaMemset(d_root_index_, 0x00, sizeof(size_type)));
    root_index_ = std::shared_ptr<size_type>(d_root_index_, cuda_deleter<size_type>());
    initialize();
  }

  void initialize() {
    const uint32_t num_blocks = 1;
    const uint32_t block_size = branching_factor;
    kernels::initialize_kernel<<<num_blocks, block_size>>>(*this);
    cuda_try(cudaDeviceSynchronize());
  }

  std::shared_ptr<size_type> root_index_;

  size_type* d_snapshot_index_;

  pair_type* h_btree_;
  size_type* h_node_count_;

  size_type* d_root_index_;

  allocator_type allocator_;
};  // namespace GpuBTree
}  // namespace GpuBTree
