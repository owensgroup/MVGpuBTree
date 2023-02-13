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

#define _CG_ABI_EXPERIMENTAL
#include <cooperative_groups.h>

#include <memory_reclaimer.hpp>

//#define DEBUG_CONCURRENT_KERNELS_OPS
// #define DEBUG_QS_ENTER_EXIST

namespace GpuBTree {
namespace kernels {

namespace cg = cooperative_groups;

template <typename key_type, typename value_type, typename size_type, typename btree>
__global__ void concurrent_find_erase_kernel_versioned(const key_type* find_keys,
                                                       value_type* find_result,
                                                       const size_type num_finds,
                                                       const key_type* erase_keys,
                                                       const size_type num_erasures,
                                                       btree tree) {
  // cuda grid
  auto block_id   = blockIdx.x;
  auto block_size = blockDim.x;

  // total number of operations
  const auto num_operations = num_erasures + num_finds;

  // insertion to operations ratio
  float erasure_ratio = static_cast<float>(num_erasures) / static_cast<float>(num_operations);
  // number of blocks that will exclusively perform insertion or ranges
  unsigned erasure_blocks = erasure_ratio * gridDim.x;
  unsigned find_blocks    = gridDim.x - erasure_blocks;

  // total number of required blocks and threads to perform all operations
  unsigned required_erasure_blocks = (num_erasures + block_size - 1) / block_size;
  unsigned required_find_blocks    = (num_finds + block_size - 1) / block_size;

  // operations tile
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<btree::branching_factor>(block);

  // allocator
  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  // reclaimer
  // using reclaimer_type                     = dummy_reclaimer;
  using reclaimer_type                     = smr::DEBR_device<btree::reclaimer_max_ptrs_count_>;
  static constexpr uint32_t required_shmem = reclaimer_type::compute_shmem_requirements();
  __shared__ uint32_t buffer_buffer[required_shmem];

  // reclaimer tile
  __shared__ cg::experimental::block_tile_memory<4, btree::reclaimer_block_size_> block_tile_shemm;
  auto thb             = cg::experimental::this_thread_block(block_tile_shemm);
  auto block_wide_tile = cg::experimental::tiled_partition<btree::reclaimer_block_size_>(thb);
  auto reclaimer =
      reclaimer_type{tree.host_reclaimer_, &buffer_buffer[0], gridDim.x, block_wide_tile};

  if (block_id < erasure_blocks) {  // first blocks will do insertion

    // iterate to perform insertions
    for (auto thread_id = threadIdx.x + block_id * block_size;
         thread_id < required_erasure_blocks * block_size;
         thread_id += (block_size * erasure_blocks)) {
      auto key      = btree::invalid_key;
      bool to_erase = false;

      if (thread_id < num_erasures) {
        key      = erase_keys[thread_id];
        to_erase = true;
      }

      reclaimer.leave_qstate(block_wide_tile, blockIdx.x, allocator, tree.allocator_);

      auto work_queue = tile.ballot(to_erase);
      while (work_queue) {
        auto cur_rank = __ffs(work_queue) - 1;
        auto cur_key  = tile.shfl(key, cur_rank);

        /*bool success = */
        tree.cooperative_erase(cur_key, tile, allocator, reclaimer);

        if (tile.thread_rank() == cur_rank) { to_erase = false; }
        work_queue = tile.ballot(to_erase);
      }

      reclaimer.enter_qstate(block_wide_tile, blockIdx.x);
    }
  } else {                       // remaining blocks will do rq
    block_id -= erasure_blocks;  // first block starts at zero

    // iterate to perform finds
    for (auto thread_id = threadIdx.x + block_id * block_size;
         thread_id < required_find_blocks * block_size;
         thread_id += (block_size * find_blocks)) {
      auto key     = btree::invalid_key;
      auto result  = btree::invalid_key;
      bool to_find = false;

      if (thread_id < num_finds) {
        key     = find_keys[thread_id];
        to_find = true;
      }

      auto work_queue = tile.ballot(to_find);
      reclaimer.leave_qstate(block_wide_tile, blockIdx.x, allocator, tree.allocator_);

      while (work_queue) {
        auto cur_rank = __ffs(work_queue) - 1;
        auto cur_key  = tile.shfl(key, cur_rank);

        auto cur_result = tree.cooperative_find(cur_key, tile, allocator, true);

        if (cur_rank == tile.thread_rank()) {
          to_find = false;
          result  = cur_result;
        }
        work_queue = tile.ballot(to_find);
      }

      reclaimer.enter_qstate(block_wide_tile, blockIdx.x);

      if (thread_id < num_finds) { find_result[thread_id] = result; }
    }
  }
}

template <typename key_type,
          typename value_type,
          typename pair_type,
          typename size_type,
          typename btree>
__global__ void concurrent_insert_range_kernel(const key_type* keys,
                                               const value_type* values,
                                               const size_type num_insertions,
                                               const key_type* lower_bounds,
                                               const key_type* upper_bounds,
                                               const size_type num_ranges,
                                               pair_type* range_result,
                                               const size_type average_range_length,
                                               btree tree) {
  // cuda grid
  auto block_id   = blockIdx.x;
  auto block_size = blockDim.x;

  // total number of operations
  const auto num_operations = num_insertions + num_ranges;

  // insertion to operations ratio
  float insertion_ratio =
      static_cast<float>(num_insertions) / static_cast<float>(num_insertions + num_ranges);
  // number of blocks that will exclusively perform insertion or ranges
  unsigned insertion_blocks = insertion_ratio * gridDim.x;
  unsigned ranges_blocks    = gridDim.x - insertion_blocks;

  // total number of required blocks and threads to perform all operations
  unsigned required_insertion_blocks = (num_insertions + block_size - 1) / block_size;
  unsigned required_ranges_blocks    = (num_ranges + block_size - 1) / block_size;
  // unsigned required_blocks = required_insertion_blocks + required_ranges_blocks;

  // operations tile
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<btree::branching_factor>(block);

  // allocator
  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  // reclaimer
  using reclaimer_type                     = smr::DEBR_device<btree::reclaimer_max_ptrs_count_>;
  static constexpr uint32_t required_shmem = reclaimer_type::compute_shmem_requirements();
  __shared__ uint32_t buffer_buffer[required_shmem];

  // reclaimer tile
  __shared__ cg::experimental::block_tile_memory<4, btree::reclaimer_block_size_> block_tile_shemm;
  auto thb             = cg::experimental::this_thread_block(block_tile_shemm);
  auto block_wide_tile = cg::experimental::tiled_partition<btree::reclaimer_block_size_>(thb);
  auto reclaimer =
      reclaimer_type{tree.host_reclaimer_, &buffer_buffer[0], gridDim.x, block_wide_tile};

  if (block_id < insertion_blocks) {  // first blocks will do insertion

    // iterate to perform insertions
    for (auto thread_id = threadIdx.x + block_id * block_size;
         thread_id < required_insertion_blocks * block_size;
         thread_id += (block_size * insertion_blocks)) {
      auto key       = btree::invalid_key;
      auto value     = btree::invalid_value;
      bool to_insert = false;

      if (thread_id < num_insertions) {
        key       = keys[thread_id];
        value     = values[thread_id];
        to_insert = true;
#ifdef DEBUG_CONCURRENT_KERNELS_OPS
        printf("global(%i, %i) local(%i, %i): inserting %i\n",
               thread_id,
               thread_id / block_size,
               threadIdx.x + blockIdx.x * blockDim.x,
               blockIdx.x,
               key);
#endif
      }

      size_type num_inserted = 1;
      auto work_queue        = tile.ballot(to_insert);
      reclaimer.leave_qstate(block_wide_tile, blockIdx.x, allocator, tree.allocator_);
#ifdef DEBUG_QS_ENTER_EXIST
      if (block_wide_tile.thread_rank() == 0) { printf("insert blk %i lqs\n", blockIdx.x); }
#endif
      while (work_queue) {
        auto cur_rank  = __ffs(work_queue) - 1;
        auto cur_key   = tile.shfl(key, cur_rank);
        auto cur_value = tile.shfl(value, cur_rank);

        /*bool success = */
        tree.cooperative_insert_versioned_out_of_place(
            cur_key, cur_value, tile, allocator, reclaimer);

        if (tile.thread_rank() == cur_rank) { to_insert = false; }
        num_inserted++;
        work_queue = tile.ballot(to_insert);
      }  // end while
      reclaimer.enter_qstate(block_wide_tile, blockIdx.x);
#ifdef DEBUG_QS_ENTER_EXIST
      if (block_wide_tile.thread_rank() == 0) { printf("insert blk %i eqs\n", blockIdx.x); }
#endif
    }
  } else {                         // remaining blocks will do rq
    block_id -= insertion_blocks;  // first block starts at zero

    // iterate to perform rqs
    for (auto thread_id = threadIdx.x + block_id * block_size;
         thread_id < required_ranges_blocks * block_size;
         thread_id += (block_size * ranges_blocks)) {
      auto lower_bound = btree::invalid_key;
      auto upper_bound = btree::invalid_key;
      bool to_find     = false;
      if (thread_id < num_ranges) {
        lower_bound = lower_bounds[thread_id];
        upper_bound = upper_bounds[thread_id];
        to_find     = true;
      }

#ifdef DEBUG_CONCURRENT_KERNELS_OPS
      printf("global(%i, %i) local(%i, %i): RQ %i\n",
             thread_id,
             thread_id / block_size,
             threadIdx.x + blockIdx.x * blockDim.x,
             blockIdx.x,
             lower_bound);
#endif

      auto tile_id              = thread_id / btree::branching_factor;
      auto first_tile_thread_id = tile_id * btree::branching_factor;

      auto work_queue = tile.ballot(to_find);
      uint32_t timestamp;  // one timestamp per find tile

      reclaimer.leave_qstate(block_wide_tile, blockIdx.x, allocator, tree.allocator_);
#ifdef DEBUG_QS_ENTER_EXIST
      if (block_wide_tile.thread_rank() == 0) { printf("query blk %i lqs\n", blockIdx.x); }
#endif
      if (work_queue) { timestamp = tree.take_snapshot(tile); }
      while (work_queue) {
        auto cur_rank        = __ffs(work_queue) - 1;
        auto cur_lower_bound = tile.shfl(lower_bound, cur_rank);
        auto cur_upper_bound = tile.shfl(upper_bound, cur_rank);

        auto offset = (first_tile_thread_id + cur_rank) * average_range_length;

        auto cur_result = tree.concurrent_cooperative_range_query(
            cur_lower_bound,
            cur_upper_bound,
            tile,
            allocator,
            timestamp,
            range_result == nullptr ? nullptr : range_result + offset);

        if (cur_rank == tile.thread_rank()) { to_find = false; }
        work_queue = tile.ballot(to_find);
      }  // end while
      reclaimer.enter_qstate(block_wide_tile, blockIdx.x);
#ifdef DEBUG_QS_ENTER_EXIST
      if (block_wide_tile.thread_rank() == 0) { printf("query blk %i eqs\n", blockIdx.x); }
#endif
    }
  }
}

template <typename key_type, typename pair_type, typename size_type, typename btree>
__global__ void range_query_kernel(const key_type* lower_bounds,
                                   const key_type* upper_bounds,
                                   pair_type* range_result,
                                   const size_type average_range_length,
                                   const size_type keys_count,
                                   btree tree,
                                   size_type* counts,
                                   const size_type timestamp,
                                   bool concurrent = false) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  auto block                = cg::this_thread_block();
  auto tile                 = cg::tiled_partition<btree::branching_factor>(block);
  auto tile_id              = thread_id / btree::branching_factor;
  auto first_tile_thread_id = tile_id * btree::branching_factor;
  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto lower_bound = btree::invalid_key;
  auto upper_bound = btree::invalid_key;

  bool to_find    = false;
  size_type count = 0;
  if (thread_id < keys_count) {
    lower_bound = lower_bounds[thread_id];
    upper_bound = upper_bounds[thread_id];
    to_find     = true;
  }

  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  auto work_queue = tile.ballot(to_find);
  while (work_queue) {
    auto cur_rank        = __ffs(work_queue) - 1;
    auto cur_lower_bound = tile.shfl(lower_bound, cur_rank);
    auto cur_upper_bound = tile.shfl(upper_bound, cur_rank);
    auto offset          = (first_tile_thread_id + cur_rank) * average_range_length;
    auto cur_result      = tree.concurrent_cooperative_range_query(
        cur_lower_bound,
        cur_upper_bound,
        tile,
        allocator,
        timestamp,
        range_result == nullptr ? nullptr : range_result + offset);
    if (cur_rank == tile.thread_rank()) {
      count   = cur_result;
      to_find = false;
    }
    work_queue = tile.ballot(to_find);
  }

  if (counts != nullptr && thread_id < keys_count) { counts[thread_id] = count; }
}

template <typename key_type, typename value_type, typename size_type, typename btree>
__global__ void insert_in_place_kernel(const key_type* keys,
                                       const value_type* values,
                                       const size_type keys_count,
                                       btree tree) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block     = cg::this_thread_block();
  auto tile      = cg::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key       = btree::invalid_key;
  auto value     = btree::invalid_value;
  bool to_insert = false;
  if (thread_id < keys_count) {
    key       = keys[thread_id];
    value     = values[thread_id];
    to_insert = true;
  }
  using allocator_type = typename btree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  size_type num_inserted = 1;
  auto work_queue        = tile.ballot(to_insert);
  while (work_queue) {
    auto cur_rank  = __ffs(work_queue) - 1;
    auto cur_key   = tile.shfl(key, cur_rank);
    auto cur_value = tile.shfl(value, cur_rank);

    /*bool success = */
    tree.cooperative_insert_versioned_in_place(cur_key, cur_value, tile, allocator);

    if (tile.thread_rank() == cur_rank) { to_insert = false; }
    num_inserted++;
    work_queue = tile.ballot(to_insert);
  }
}

template <typename key_type, typename value_type, typename size_type, typename btree>
__global__ void insert_out_of_place_kernel(const key_type* keys,
                                           const value_type* values,
                                           const size_type num_insertions,
                                           btree tree) {
  auto block                         = cg::this_thread_block();
  auto tile                          = cg::tiled_partition<btree::branching_factor>(block);
  auto block_id                      = blockIdx.x;
  auto block_size                    = blockDim.x;
  unsigned insertion_blocks          = gridDim.x;
  unsigned required_insertion_blocks = (num_insertions + block_size - 1) / block_size;

  using allocator_type = typename btree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  // reclaimer
  dummy_reclaimer reclaimer;

  // using reclaimer_type = smr::DEBR_device<btree::reclaimer_max_ptrs_count_>;
  // static constexpr uint32_t required_shmem =
  // reclaimer_type::compute_shmem_requirements();
  // __shared__ uint32_t buffer_buffer[required_shmem];
  // auto reclaimer = reclaimer_type{tree.host_reclaimer_, &buffer_buffer[0], gridDim.x};

  // reclaimer tile
  // __shared__ cg::experimental::block_tile_memory<4, btree::reclaimer_block_size_>
  //     block_tile_shemm;
  // auto thb = cg::experimental::this_thread_block(block_tile_shemm);
  // auto block_wide_tile =
  //     cg::experimental::tiled_partition<btree::reclaimer_block_size_>(thb);

  // iterate to perform insertions
  for (auto thread_id = threadIdx.x + block_id * block_size;
       thread_id < required_insertion_blocks * block_size;
       thread_id += (block_size * insertion_blocks)) {
    auto key       = btree::invalid_key;
    auto value     = btree::invalid_value;
    bool to_insert = false;

    if (thread_id < num_insertions) {
      key       = keys[thread_id];
      value     = values[thread_id];
      to_insert = true;
    }

    size_type num_inserted = 1;
    auto work_queue        = tile.ballot(to_insert);
    // we are not going to reclaim memory here since future queries may use old snapshots
    // we are still storing pointers somewhere (depending on the configuration of the
    // reclaimer)... if reclamation is required then we should store all retired pointers
    // into global memory and retire them later
    // note: the presistent kernel style and code used here are not suitable for the case
    // above because we overwrite pointers. so, the leave/enter qstate must move outside
    // the loop

    // reclaimer.leave_qstate(block_wide_tile, blockIdx.x, allocator, tree.allocator_);
    while (work_queue) {
      auto cur_rank  = __ffs(work_queue) - 1;
      auto cur_key   = tile.shfl(key, cur_rank);
      auto cur_value = tile.shfl(value, cur_rank);

      /*bool success = */
      tree.cooperative_insert_versioned_out_of_place(
          cur_key, cur_value, tile, allocator, reclaimer);

      if (tile.thread_rank() == cur_rank) { to_insert = false; }
      num_inserted++;
      work_queue = tile.ballot(to_insert);
    }
    // reclaimer.enter_qstate(block_wide_tile, blockIdx.x);
  }
}
template <typename btree>
__global__ void take_snapshot_kernel(btree tree) {
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<btree::branching_factor>(block);
  tree.take_snapshot(tile);
}

template <typename key_type, typename value_type, typename size_type, typename btree>
__global__ void find_kernel(const key_type* keys,
                            value_type* values,
                            const size_type keys_count,
                            btree tree,
                            const size_type time_stamp,
                            bool concurrent = false) {
  using pair_type = typename btree::pair_type;

  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key     = btree::invalid_key;
  auto value   = btree::invalid_value;
  bool to_find = false;
  if (thread_id < keys_count) {
    key     = keys[thread_id];
    to_find = true;
  }

  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  auto work_queue = tile.ballot(to_find);
  while (work_queue) {
    auto cur_rank   = __ffs(work_queue) - 1;
    auto cur_key    = tile.shfl(key, cur_rank);
    auto cur_result = tree.cooperative_find(cur_key, tile, allocator, time_stamp, concurrent);
    if (cur_rank == tile.thread_rank()) {
      value   = cur_result;
      to_find = false;
    }
    work_queue = tile.ballot(to_find);
  }

  if (thread_id < keys_count) { values[thread_id] = value; }
}

/********************************/
/*        Common kernels       */
/******************************/
template <typename btree>
__global__ void initialize_kernel(btree tree) {
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<btree::branching_factor>(block);

  using allocator_type = typename btree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  tree.allocate_root_node(tile, allocator);
}
template <typename key_type, typename value_type, typename size_type, typename btree>
__global__ void find_kernel(const key_type* keys,
                            value_type* values,
                            const size_type keys_count,
                            btree tree,
                            bool concurrent = false) {
  using pair_type = typename btree::pair_type;
  auto thread_id  = threadIdx.x + blockIdx.x * blockDim.x;

  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key     = btree::invalid_key;
  auto value   = btree::invalid_value;
  bool to_find = false;
  if (thread_id < keys_count) {
    key     = keys[thread_id];
    to_find = true;
  }

  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  auto work_queue = tile.ballot(to_find);
  while (work_queue) {
    auto cur_rank = __ffs(work_queue) - 1;
    auto cur_key  = tile.shfl(key, cur_rank);
    value_type cur_result;
    cur_result = tree.cooperative_find(cur_key, tile, allocator, concurrent);
    if (cur_rank == tile.thread_rank()) {
      value   = cur_result;
      to_find = false;
    }
    work_queue = tile.ballot(to_find);
  }

  if (thread_id < keys_count) { values[thread_id] = value; }
}
template <typename key_type, typename pair_type, typename size_type, typename btree>
__global__ void range_query_kernel(const key_type* lower_bounds,
                                   const key_type* upper_bounds,
                                   pair_type* result,
                                   const size_type average_range_length,
                                   const size_type keys_count,
                                   btree tree,
                                   size_type* counts,
                                   bool concurrent = false) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  auto block                = cg::this_thread_block();
  auto tile                 = cg::tiled_partition<btree::branching_factor>(block);
  auto tile_id              = thread_id / btree::branching_factor;
  auto first_tile_thread_id = tile_id * btree::branching_factor;
  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto lower_bound = btree::invalid_key;
  auto upper_bound = btree::invalid_key;

  bool to_find    = false;
  size_type count = 0;
  if (thread_id < keys_count) {
    lower_bound = lower_bounds[thread_id];
    upper_bound = upper_bounds[thread_id];
    to_find     = true;
  }

  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  auto work_queue = tile.ballot(to_find);
  while (work_queue) {
    auto cur_rank        = __ffs(work_queue) - 1;
    auto cur_lower_bound = tile.shfl(lower_bound, cur_rank);
    auto cur_upper_bound = tile.shfl(upper_bound, cur_rank);
    auto offset          = (first_tile_thread_id + cur_rank) * average_range_length;
    auto cur_result      = tree.cooperative_range_query(cur_lower_bound,
                                                   cur_upper_bound,
                                                   tile,
                                                   allocator,
                                                   result == nullptr ? nullptr : result + offset,
                                                   concurrent);
    if (cur_rank == tile.thread_rank()) {
      count   = cur_result;
      to_find = false;
    }
    work_queue = tile.ballot(to_find);
  }

  if (counts != nullptr && thread_id < keys_count) { counts[thread_id] = count; }
}

template <typename key_type, typename size_type, typename btree>
__global__ void erase_kernel(const key_type* keys,
                             const size_type keys_count,
                             btree tree,
                             bool concurrent = false) {
  using pair_type = typename btree::pair_type;

  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key             = btree::invalid_key;
  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  bool to_erase = false;
  // bool erased = false;
  if (thread_id < keys_count) {
    key      = keys[thread_id];
    to_erase = true;
  }
  auto work_queue = tile.ballot(to_erase);
  while (work_queue) {
    auto cur_rank = __ffs(work_queue) - 1;
    auto cur_key  = tile.shfl(key, cur_rank);
    // auto cur_result =
    tree.cooperative_erase(cur_key, tile, allocator, concurrent);
    if (cur_rank == tile.thread_rank()) {
      to_erase = false;
      // erased = cur_result;
    }
    work_queue = tile.ballot(to_erase);
  }
  // (void)erased;
}

/*
B-link-Tree kernels
*/

template <typename key_type, typename value_type, typename size_type, typename btree>
__global__ void concurrent_find_erase_kernel_blink(const key_type* find_keys,
                                                   value_type* find_result,
                                                   const size_type num_finds,
                                                   const key_type* erase_keys,
                                                   const size_type num_erasures,
                                                   btree tree) {
  // cuda grid
  auto block_id   = blockIdx.x;
  auto block_size = blockDim.x;

  // total number of operations
  const auto num_operations = num_erasures + num_finds;

  // insertion to operations ratio
  float erasure_ratio = static_cast<float>(num_erasures) / static_cast<float>(num_operations);
  // number of blocks that will exclusively perform insertion or ranges
  unsigned erasure_blocks = erasure_ratio * gridDim.x;
  unsigned find_blocks    = gridDim.x - erasure_blocks;

  // total number of required blocks and threads to perform all operations
  unsigned required_erasure_blocks = (num_erasures + block_size - 1) / block_size;
  unsigned required_find_blocks    = (num_finds + block_size - 1) / block_size;

  // operations tile
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<btree::branching_factor>(block);

  // allocator
  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  if (block_id < erasure_blocks) {  // first blocks will do insertion

    // iterate to perform insertions
    for (auto thread_id = threadIdx.x + block_id * block_size;
         thread_id < required_erasure_blocks * block_size;
         thread_id += (block_size * erasure_blocks)) {
      auto key      = btree::invalid_key;
      bool to_erase = false;
      if (thread_id < num_erasures) {
        key      = erase_keys[thread_id];
        to_erase = true;
      }

      auto work_queue = tile.ballot(to_erase);
      while (work_queue) {
        auto cur_rank = __ffs(work_queue) - 1;
        auto cur_key  = tile.shfl(key, cur_rank);

        /*bool success = */
        tree.cooperative_erase(cur_key, tile, allocator, true);

        if (tile.thread_rank() == cur_rank) { to_erase = false; }
        work_queue = tile.ballot(to_erase);
      }
    }
  } else {                       // remaining blocks will do rq
    block_id -= erasure_blocks;  // first block starts at zero

    // iterate to perform finds
    for (auto thread_id = threadIdx.x + block_id * block_size;
         thread_id < required_find_blocks * block_size;
         thread_id += (block_size * find_blocks)) {
      auto key     = btree::invalid_key;
      auto result  = btree::invalid_key;
      bool to_find = false;
      if (thread_id < num_finds) {
        key     = find_keys[thread_id];
        to_find = true;
      }

      auto work_queue = tile.ballot(to_find);

      while (work_queue) {
        auto cur_rank = __ffs(work_queue) - 1;
        auto cur_key  = tile.shfl(key, cur_rank);

        auto cur_result = tree.cooperative_find(cur_key, tile, allocator, true);

        if (cur_rank == tile.thread_rank()) {
          to_find = false;
          result  = cur_result;
        }
        work_queue = tile.ballot(to_find);
      }
      if (thread_id < num_finds) { find_result[thread_id] = result; }
    }
  }
}

template <typename key_type,
          typename value_type,
          typename pair_type,
          typename size_type,
          typename btree>
__global__ void concurrent_insert_range_kernel_blink(const key_type* keys,
                                                     const value_type* values,
                                                     const size_type num_insertions,
                                                     const key_type* lower_bounds,
                                                     const key_type* upper_bounds,
                                                     const size_type num_ranges,
                                                     pair_type* range_result,
                                                     const size_type average_range_length,
                                                     btree tree) {
  // cuda grid
  auto block_id   = blockIdx.x;
  auto block_size = blockDim.x;

  // total number of operations
  const auto num_operations = num_insertions + num_ranges;

  // insertion to operations ratio
  float insertion_ratio =
      static_cast<float>(num_insertions) / static_cast<float>(num_insertions + num_ranges);
  // number of blocks that will exclusively perform insertion or ranges
  unsigned insertion_blocks = insertion_ratio * gridDim.x;
  unsigned ranges_blocks    = gridDim.x - insertion_blocks;

  // total number of required blocks and threads to perform all operations
  unsigned required_insertion_blocks = (num_insertions + block_size - 1) / block_size;
  unsigned required_ranges_blocks    = (num_ranges + block_size - 1) / block_size;
  // unsigned required_blocks = required_insertion_blocks + required_ranges_blocks;

  // operations tile
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<btree::branching_factor>(block);

  // allocator
  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  if (block_id < insertion_blocks) {  // first blocks will do insertion

    // iterate to perform insertions
    for (auto thread_id = threadIdx.x + block_id * block_size;
         thread_id < required_insertion_blocks * block_size;
         thread_id += (block_size * insertion_blocks)) {
      auto key       = btree::invalid_key;
      auto value     = btree::invalid_value;
      bool to_insert = false;
      if (thread_id < num_insertions) {
        key       = keys[thread_id];
        value     = values[thread_id];
        to_insert = true;
      }

      size_type num_inserted = 1;
      auto work_queue        = tile.ballot(to_insert);
      while (work_queue) {
        auto cur_rank  = __ffs(work_queue) - 1;
        auto cur_key   = tile.shfl(key, cur_rank);
        auto cur_value = tile.shfl(value, cur_rank);

        /*bool success = */
        tree.cooperative_insert(cur_key, cur_value, tile, allocator);

        if (tile.thread_rank() == cur_rank) { to_insert = false; }
        num_inserted++;
        work_queue = tile.ballot(to_insert);
      }
    }
  } else {                         // remaining blocks will do rq
    block_id -= insertion_blocks;  // first block starts at zero

    // iterate to perform rqs
    for (auto thread_id = threadIdx.x + block_id * block_size;
         thread_id < required_ranges_blocks * block_size;
         thread_id += (block_size * ranges_blocks)) {
      auto lower_bound = btree::invalid_key;
      auto upper_bound = btree::invalid_key;
      bool to_find     = false;

      if (thread_id < num_ranges) {
        lower_bound = lower_bounds[thread_id];
        upper_bound = upper_bounds[thread_id];
        to_find     = true;
      }

      auto tile_id              = thread_id / btree::branching_factor;
      auto first_tile_thread_id = tile_id * btree::branching_factor;

      auto work_queue = tile.ballot(to_find);

      while (work_queue) {
        auto cur_rank        = __ffs(work_queue) - 1;
        auto cur_lower_bound = tile.shfl(lower_bound, cur_rank);
        auto cur_upper_bound = tile.shfl(upper_bound, cur_rank);

        auto offset = (first_tile_thread_id + cur_rank) * average_range_length;

        auto cur_result =
            tree.cooperative_range_query(cur_lower_bound,
                                         cur_upper_bound,
                                         tile,
                                         allocator,
                                         range_result == nullptr ? nullptr : range_result + offset,
                                         true);

        if (cur_rank == tile.thread_rank()) { to_find = false; }
        work_queue = tile.ballot(to_find);
      }
    }
  }
}

template <typename key_type, typename value_type, typename size_type, typename btree>
__global__ void insert_kernel(const key_type* keys,
                              const value_type* values,
                              const size_type keys_count,
                              btree tree) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block     = cg::this_thread_block();
  auto tile      = cg::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key       = btree::invalid_key;
  auto value     = btree::invalid_value;
  bool to_insert = false;
  if (thread_id < keys_count) {
    key       = keys[thread_id];
    value     = values[thread_id];
    to_insert = true;
  }
  using allocator_type = typename btree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  size_type num_inserted = 1;
  auto work_queue        = tile.ballot(to_insert);
  while (work_queue) {
    auto cur_rank  = __ffs(work_queue) - 1;
    auto cur_key   = tile.shfl(key, cur_rank);
    auto cur_value = tile.shfl(value, cur_rank);

    /*bool success = */
    tree.cooperative_insert(cur_key, cur_value, tile, allocator);

    if (tile.thread_rank() == cur_rank) { to_insert = false; }
    num_inserted++;
    work_queue = tile.ballot(to_insert);
  }
}

__device__ int get_node_height(uint32_t& node_idx,
                               uint32_t num_leaves,
                               uint32_t n,
                               uint32_t b,
                               bool& not_last) {
  for (int level = 0;; level++) {
    if (node_idx < num_leaves) {
      not_last = (node_idx != (num_leaves - 1));
      return level;
    }
    node_idx -= num_leaves;
    int frac = (num_leaves % b) ? 1 : 0;
    num_leaves >>= n;
    num_leaves += frac;
  }
}

template <typename key_type, typename value_type, typename btree>
__global__ void bulk_build_kernel(const key_type* keys,
                                  const value_type* values,
                                  const uint32_t num_keys,
                                  const uint32_t num_nodes,
                                  const uint32_t num_leaves,
                                  const uint32_t tree_height,
                                  const uint32_t bulk_build_branching_factor,
                                  const uint32_t log_bulk_build_branching_factor,
                                  btree tree) {
  // TODO: Is key 0 still reserved?
  uint32_t tid      = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t lane_idx = threadIdx.x & 0x1F;
  uint32_t node_idx = tid / 32;

  if (node_idx >= num_nodes) return;

  uint32_t local_node_idx = node_idx;
  bool not_last_node      = false;

  uint32_t node_height = get_node_height(local_node_idx,
                                         num_leaves,
                                         log_bulk_build_branching_factor,
                                         bulk_build_branching_factor,
                                         not_last_node);

  uint32_t hB = powf(bulk_build_branching_factor, node_height);

  uint32_t lane_data = btree::invalid_key;

  if (node_height > 0) {
    uint32_t first_node = 0;
    uint32_t num_lev    = num_leaves;
    for (int i = 0; i < node_height - 1; i++) {
      first_node += num_lev;
      int frac = (num_lev % bulk_build_branching_factor) ? 1 : 0;
      num_lev >>= log_bulk_build_branching_factor;
      num_lev += frac;
    }
    first_node++;
    uint32_t child_idx = first_node + local_node_idx * bulk_build_branching_factor;
    if (lane_idx < 16) {
      uint32_t to_read = local_node_idx * hB * bulk_build_branching_factor + (lane_idx / 2) * hB;
      if (to_read < num_keys) {
        child_idx = (child_idx + (lane_idx / 2));
        if (to_read == 0) {
          lane_data = (lane_idx % 2) ? child_idx : 0;
        } else {
          lane_data = (lane_idx % 2) ? child_idx : keys[to_read - 1];
        }
        lane_data = lane_data;
      }
    }
  } else {
    if (lane_idx < 16) {
      uint32_t to_read = local_node_idx * hB * bulk_build_branching_factor + (lane_idx / 2) * hB;
      if (to_read < num_keys) {
        if (to_read == 0) {
          lane_data = (lane_idx % 2) ? 0 : 0;
        } else {
          lane_data = (lane_idx % 2) ? values[to_read - 1] : keys[to_read - 1];
        }
      }
    }
  }
  node_idx++;
  if (not_last_node) {
    lane_data = (lane_idx == 31) ? node_idx + 1 : lane_data;
    lane_data = (lane_idx == 30) ? keys[(local_node_idx + 1) * hB * bulk_build_branching_factor - 1]
                                 : lane_data;
  }

  node_idx = (node_idx == num_nodes) ? 0 : node_idx;

  const uint32_t bits_per_byte   = 8;
  const uint32_t lock_bit_offset = sizeof(key_type) * bits_per_byte - 1;
  const uint32_t leaf_bit_offset = lock_bit_offset - 1;
  const uint32_t lock_bit_mask   = 1u << lock_bit_offset;
  const uint32_t leaf_bit_mask   = 1u << leaf_bit_offset;

  if (lane_idx == 31 && node_height == 0) {
    lane_data = lane_data | leaf_bit_mask;     // set_leaf_bit
    lane_data = lane_data & (~lock_bit_mask);  // unset_lock_bit
  }
  if (lane_idx == 31 && node_height != 0) {
    lane_data = lane_data & (~leaf_bit_mask);  // unset_leaf_bit
    lane_data = lane_data & (~lock_bit_mask);  // unset_lock_bit
  }
  using allocator_type      = typename btree::device_allocator_context_type;
  uint32_t* raw_tree_buffer = reinterpret_cast<uint32_t*>(tree.allocator_.get_raw_buffer());

  raw_tree_buffer[node_idx * btree::branching_factor * 2 + lane_idx] = lane_data;
  if (tid == 0) { tree.allocator_.set_allocated_count(num_nodes); }
}

}  // namespace kernels
}  // namespace GpuBTree
