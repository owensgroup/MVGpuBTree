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
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <host_allocators.hpp>

#include <device_bump_allocator.hpp>
#include <memory>
#include <memory_utils.hpp>

// DEBR implemenation based on:
// Reclaiming Memory for Lock-Free Data Structures: There has to be a Better Way
// By Trevor Brown

// #define DEBUG_PRINTF
// #define DEBUG_FREE
// #define DEBUG_RETIRE
// #define DEBUG_SUMMARY_PRINTF

// #define COLLECT_RECLAIMER_STATS
#ifdef COLLECT_RECLAIMER_STATS
#define COLLECT_RECLAIMER_STATS_MAX_EPOCHS 2048
#else
#define COLLECT_RECLAIMER_STATS_MAX_EPOCHS 0
#endif
struct dummy_reclaimer {
  using address_type = uint32_t;
  template <typename tile_type, typename Allocator, typename AllocatorCtx>
  __device__ void retire(const address_type& address,
                         tile_type& tile,
                         Allocator& allocator,
                         AllocatorCtx& alloc_ctx) {}
  __device__ __host__ static constexpr uint32_t compute_shmem_requirements() { return 1; }

  template <typename tile_type, typename Allocator, typename AllocatorCtx>
  DEVICE_QUALIFIER void leave_qstate(tile_type& tile,
                                     int block_id,
                                     Allocator& allocator,
                                     AllocatorCtx& alloc_ctx) {}
  template <typename tile_type>
  DEVICE_QUALIFIER void enter_qstate(tile_type& tile, int block_id) {}

  template <typename tile_type, typename host_reclaimer_type>
  DEVICE_QUALIFIER dummy_reclaimer(const host_reclaimer_type& host,
                                   uint32_t* shared_buffer,
                                   uint32_t num_active_blocks,
                                   const tile_type& tile){};
  DEVICE_QUALIFIER dummy_reclaimer(){};
};

namespace smr {
using address_type = uint32_t;

struct DEBR_host {
  // Destructor that frees all memory
  HOST_DEVICE_QUALIFIER ~DEBR_host() {}

  // carefully written copy constructor
  DEBR_host(const DEBR_host& other)
      : announce_(other.announce_)
      , current_epoch_(other.current_epoch_)
      , d_announce_(other.d_announce_)
      , d_current_epoch_(other.d_current_epoch_)
      , num_blocks_(other.num_blocks_)
      , external_buffer_size_(other.external_buffer_size_) {}

  DEBR_host(uint32_t num_blocks, uint32_t external_buffer_size = 10'000'000)
      : num_blocks_(num_blocks), external_buffer_size_(external_buffer_size) {
    allocate(num_blocks, external_buffer_size);
  }
  DEBR_host() = delete;

  __host__ void allocate(std::size_t num_blocks, std::size_t buffer_size) {
    d_announce_      = cuda_allocator<uint32_t>().allocate(num_blocks + buffer_size +
                                                      COLLECT_RECLAIMER_STATS_MAX_EPOCHS * 2);
    d_current_epoch_ = cuda_allocator<uint32_t>().allocate(1);

    cuda_try(cudaMemset(d_current_epoch_, 0x00, sizeof(uint32_t)));

    thrust::fill(thrust::device, d_announce_, d_announce_ + num_blocks, 0x00000001);
    if (buffer_size != 0) {
      thrust::fill(thrust::device,
                   d_announce_ + num_blocks,
                   d_announce_ + num_blocks + buffer_size,
                   0xffffffff);
    }
    if (COLLECT_RECLAIMER_STATS_MAX_EPOCHS != 0) {
      thrust::fill(thrust::device,
                   d_announce_ + num_blocks + buffer_size,
                   d_announce_ + num_blocks + buffer_size + COLLECT_RECLAIMER_STATS_MAX_EPOCHS * 2,
                   0xffffffff);
    }

    announce_      = std::shared_ptr<uint32_t>(d_announce_, cuda_deleter<uint32_t>());
    current_epoch_ = std::shared_ptr<uint32_t>(d_current_epoch_, cuda_deleter<uint32_t>());
  }

  std::vector<std::pair<uint32_t, uint32_t>> compute_epochs_stats() {
#ifdef COLLECT_RECLAIMER_STATS
    std::vector<uint32_t> epoch_pointers_count(COLLECT_RECLAIMER_STATS_MAX_EPOCHS * 2);
    cuda_try(cudaMemcpy(epoch_pointers_count.data(),
                        d_announce_ + num_blocks_ + external_buffer_size_,
                        COLLECT_RECLAIMER_STATS_MAX_EPOCHS * 2 * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
    std::vector<std::pair<uint32_t, uint32_t>> reclaim_vs_none;
    uint32_t epoch_counts = 0;
    while (epoch_pointers_count[epoch_counts] != 0xffffffff) {
      reclaim_vs_none.push_back(
          {epoch_pointers_count[epoch_counts],
           epoch_pointers_count[epoch_counts + COLLECT_RECLAIMER_STATS_MAX_EPOCHS]});
      epoch_counts++;
    }
    return reclaim_vs_none;
#else
    return std::vector<std::pair<uint32_t, uint32_t>>{};
#endif
  }
  std::shared_ptr<uint32_t> announce_;
  std::shared_ptr<uint32_t> current_epoch_;
  uint32_t* d_announce_;       // an array of num_blocks values each contain epoch  number with
                               // LSB for is quiescent
  uint32_t* d_current_epoch_;  // a single number
  uint32_t num_blocks_;
  uint32_t external_buffer_size_;
};

template <uint32_t SHARED_PTRS_COUNT = 1024>
struct DEBR_device {
  template <typename tile_type, typename Allocator, typename AllocatorCtx>
  DEVICE_QUALIFIER void retire(const address_type& address,
                               tile_type& tile,
                               Allocator& allocator,
                               AllocatorCtx& alloc_ctx) {
    // add to current limbo bag
    if (tile.thread_rank() == 0) {
      auto cur_bag         = *cur_bag_;
      auto num_in_limbobag = atomicAdd((uint32_t*)count_per_bag_ + cur_bag, 1);

      // if this happens then we need to use global memory to store overflows
      if (num_in_limbobag >= PTRS_PER_BAG) {
        // location of the external bags
        auto tile_external_buffer_size   = external_buffer_size_ / num_active_bocks_;
        auto tile_external_buffer_offset = tile_external_buffer_size * blockIdx.x;

        // size of one bag
        auto bag_external_buffer_size   = tile_external_buffer_size / NUM_BAGS;
        auto bag_external_buffer_offset = bag_external_buffer_size * cur_bag;

        auto count_in_external_bag = num_in_limbobag - PTRS_PER_BAG;

#ifdef DEBUG_RETIRE
        // reclaimer out of space
        if (count_in_external_bag >= bag_external_buffer_size) {
          printf("%u is greater than %u\n", count_in_external_bag, bag_external_buffer_size);
          __threadfence_system();
        }

        cuda_assert(count_in_external_bag < bag_external_buffer_size);
#endif
        // store in slab bag
        // the offset include:
        // num_blocks: offset to skip announce data
        // tile external offset: skip other tiles
        // bag external offset: skip other bags
        // count: skip the pointers count in bag
        auto offset_in_announce = (num_blocks_ + tile_external_buffer_offset +
                                   bag_external_buffer_offset + count_in_external_bag);

#ifdef DEBUG_RETIRE
        auto old = atomicExch(announce_ + offset_in_announce, address);
        if (old != INVALID_ADDRESS) {
          printf("Overwriting %i, %i, %i, %i, %i, %i, %i, %i, %i!!\n",
                 blockIdx.x,
                 tile_external_buffer_size,
                 tile_external_buffer_offset,
                 bag_external_buffer_size,
                 bag_external_buffer_offset,
                 count_in_external_bag,
                 offset_in_announce,
                 old,
                 cur_bag);
        }
#else
        cuda_memory<address_type>::store(announce_ + offset_in_announce, address);
#endif

#if defined DEBUG_PRINTF || defined DEBUG_RETIRE
        printf("%u (global %u - blk %i) Added: %u to slab bag %u at idx %u (offset = %u)\n",
               tile.thread_rank(),
               threadIdx.x + blockIdx.x * blockDim.x,
               blockIdx.x,
               address,
               cur_bag,
               num_in_limbobag,
               offset_in_announce);
#endif
      } else {
        auto bag_offset         = cur_bag * PTRS_PER_BAG + num_in_limbobag;
        limbo_bags_[bag_offset] = address;

#if defined DEBUG_PRINTF || defined DEBUG_RETIRE
        printf("%u (global %u) Added: %u to bag %u at idx %u (offset = %u)\n",
               tile.thread_rank(),
               threadIdx.x + blockIdx.x * blockDim.x,
               address,
               cur_bag,
               num_in_limbobag,
               bag_offset);
#endif
      }
    }
  }
  DEVICE_QUALIFIER uint32_t set_quiescent_bit(uint32_t announce) {
    return announce | quiescent_bit_mask;
  }
  DEVICE_QUALIFIER bool is_quiescent(uint32_t announce) {
    return (announce & quiescent_bit_mask) == quiescent_bit_mask;
  }
  DEVICE_QUALIFIER uint32_t mask_quiescent_bit(uint32_t announce) {
    auto mask = ~quiescent_bit_mask;
    return announce & mask;
  }
  DEVICE_QUALIFIER void set_quiescent_bit(uint32_t* announce) {
    atomicOr(announce, quiescent_bit_mask);
  }

  // called before the start of data structure operation
  template <typename tile_type, typename Allocator, typename AllocatorCtx>
  DEVICE_QUALIFIER void leave_qstate(tile_type& tile,
                                     int block_id,
                                     Allocator& allocator,
                                     AllocatorCtx& alloc_ctx) {
    tile.sync();
    __threadfence();

    // read current_epoch
    uint32_t cur_e =
        cuda_memory<uint32_t>::load(current_epoch_, cuda_memory_order::memory_order_relaxed);
    auto thread_rank = tile.thread_rank();
    uint32_t old_e;
    // set announced_array
    if (thread_rank == 0) {
      // atomically unset q bit and set the epoch number
      old_e = atomicExch(announce_ + block_id, cur_e);
#ifdef DEBUG_PRINTF
      cuda_assert(is_quiescent(old_e));  // must be quiescent
#endif
      old_e = mask_quiescent_bit(old_e);
      if (old_e != cur_e) {
        *advance_epoch_ = true;  // communicate with other threads in tile
      }
    }

    // sync threads
    tile.sync();

    // if we can advance we reclaim current libmo bag and change the cur_bag
    if (*advance_epoch_) {
      auto cur_bag     = *cur_bag_;
      cur_bag          = (cur_bag + 1) % NUM_BAGS;
      auto num_to_free = count_per_bag_[cur_bag];

      // call to free
      auto stride         = tile.size();
      auto count_in_bags  = min(num_to_free, PTRS_PER_BAG);
      auto rounded_blocks = (count_in_bags + stride - 1) / stride;

      // Free everything in shemm
      for (uint32_t i = thread_rank; i < rounded_blocks * stride; i += stride) {
        address_type address_to_free = INVALID_ADDRESS;
        if (i < count_in_bags) { address_to_free = limbo_bags_[cur_bag * PTRS_PER_BAG + i]; }
        if (address_to_free != INVALID_ADDRESS) {
#if defined DEBUG_PRINTF || defined DEBUG_FREE
          printf("Block %i is freeing ptr (shmem) %u at %u\n", block_id, address_to_free, i);
#endif
          allocator.deallocate(
              alloc_ctx, address_to_free, static_cast<typename Allocator::size_type>(0));
        }
      }
      // free other pointers in gmem
      if (num_to_free >= PTRS_PER_BAG) {
        auto count_in_extern_bags = num_to_free - PTRS_PER_BAG;
        rounded_blocks            = (count_in_extern_bags + stride - 1) / stride;

        // location of the external bags
        auto tile_external_buffer_size   = external_buffer_size_ / num_active_bocks_;
        auto tile_external_buffer_offset = tile_external_buffer_size * block_id;

        // size of one bag
        auto bag_external_buffer_size   = tile_external_buffer_size / NUM_BAGS;
        auto bag_external_buffer_offset = bag_external_buffer_size * cur_bag;

        auto announce_offset =
            num_blocks_ + tile_external_buffer_offset + bag_external_buffer_offset;

        for (uint32_t i = thread_rank; i < rounded_blocks * stride; i += stride) {
          address_type address_to_free = INVALID_ADDRESS;
          if (i < count_in_extern_bags) {
#ifdef DEBUG_RETIRE
            address_to_free = atomicExch(announce_ + announce_offset + i, INVALID_ADDRESS);
#else
            address_to_free = announce_[announce_offset + i];
#endif
          }
          if (address_to_free != INVALID_ADDRESS) {
#if defined DEBUG_PRINTF || defined DEBUG_FREE
            printf("Block %i is freeing ptr (gmem) %u at (%u) %u (b=%i)\n",
                   block_id,
                   address_to_free,
                   announce_offset + i,
                   i,
                   cur_bag);
#endif
            allocator.deallocate(
                alloc_ctx, address_to_free, static_cast<typename Allocator::size_type>(0));
          }
        }
      }
#ifdef DEBUG_SUMMARY_PRINTF
      if (tile.thread_rank() == 0) {
        printf("Block %i freed %u pointers\n", block_id, num_to_free);
      }
#endif

      tile.sync();
      count_per_bag_[cur_bag] = 0;  // reset the counter
                                    // one thread writes the new epoch number
      if (thread_rank == 0) {
        *cur_bag_ = cur_bag;

#ifdef DEBUG_PRINTF
        printf("Block %i adv bag %u e %i %i\n ", block_id, cur_bag, cur_e, old_e);
#endif
      }
    }
    tile.sync();
    *advance_epoch_ = false;
    {
      auto stride         = tile.size();
      auto rounded_blocks = (num_active_bocks_ + stride - 1) / stride;

      for (uint32_t i = thread_rank; i < rounded_blocks * stride; i += stride) {
        bool p_is_quiescent = true;
        uint32_t p_epoch    = cur_e;
        if (i < num_active_bocks_) {
          p_epoch =
              cuda_memory<uint32_t>::load(&announce_[i], cuda_memory_order::memory_order_relaxed);
          p_is_quiescent = is_quiescent(p_epoch);
          p_epoch        = mask_quiescent_bit(p_epoch);
#ifdef DEBUG_PRINTF
          printf("Block %i sees block %i at epoch %u and is quiescent %i\n",
                 block_id,
                 i,
                 p_epoch,
                 p_is_quiescent);
#endif
        }

        bool advanced = p_is_quiescent || p_epoch == cur_e;
        if (!tile.all(advanced)) { return; }
      }

      // advance the epoch
      if (tile.thread_rank() == 0) {
        bool advanced_epoch_success =
            atomicCAS(current_epoch_, cur_e, cur_e + 2) == cur_e;  // epoch numbers are even

#if defined DEBUG_PRINTF || defined DEBUG_FREE
        if (advanced_epoch_success) { printf("blk %i adve %u %u\n", block_id, cur_e, cur_e + 2); }
#endif
#ifdef COLLECT_RECLAIMER_STATS
        if (advanced_epoch_success) {
          auto usage           = allocator.get_allocated_count();
          auto total_allocated = allocator.get_total_allocated_count();
          auto maximum         = allocator.get_max_allocated_count();
          auto epoch_number    = (cur_e >> 1);
          cuda_assert(epoch_number < COLLECT_RECLAIMER_STATS_MAX_EPOCHS);
          uint32_t epoch_offset   = num_blocks_ + external_buffer_size_ + epoch_number;
          announce_[epoch_offset] = usage;
          announce_[epoch_offset + COLLECT_RECLAIMER_STATS_MAX_EPOCHS] = total_allocated;
        }
#endif
      }
    }
  }

  // called after the data structure operation
  template <typename tile_type>
  DEVICE_QUALIFIER void enter_qstate(tile_type& tile, int block_id) {
    tile.sync();
    if (tile.thread_rank() == 0) { set_quiescent_bit(announce_ + block_id); }
    tile.sync();
    __threadfence();
  }

  // called at the end of a block execution to flush all bags to the external buffer in DRAM
  // todo: test flush_shmem_bags
  template <typename tile_type>
  DEVICE_QUALIFIER void flush_shmem_bags(tile_type& tile, int block_id) {
    tile.sync();
    auto stride                      = tile.size();
    auto thread_rank                 = tile.thread_rank();
    auto tile_external_buffer_size   = external_buffer_size_ / num_active_bocks_;
    auto tile_external_buffer_offset = tile_external_buffer_size * block_id;

    auto bag_external_buffer_size = tile_external_buffer_size / NUM_BAGS;
    // todo: store the total number of pointers in DRAM (currently they are only stored in shmem)

    for (int cur_bag = 0; cur_bag < NUM_BAGS; cur_bag++) {
      auto num_to_free                = count_per_bag_[cur_bag];
      auto count_in_extern_bags       = num_to_free - PTRS_PER_BAG;
      auto num_to_flush               = num_to_free - count_in_extern_bags;
      auto rounded_blocks             = (num_to_flush + stride - 1) / stride;
      auto bag_external_buffer_offset = bag_external_buffer_size * cur_bag;
      for (uint32_t i = thread_rank; i < rounded_blocks * stride; i += stride) {
        auto offset_in_announce = (num_blocks_ + tile_external_buffer_offset +
                                   bag_external_buffer_offset + count_in_extern_bags + i);
        auto bag_offset         = cur_bag * PTRS_PER_BAG + thread_rank;
        auto bag_end_offset     = (cur_bag + 1) * PTRS_PER_BAG + thread_rank;
        if (bag_offset < bag_end_offset) {
          auto address = limbo_bags_[bag_offset];
          if (address != INVALID_ADDRESS) {
            cuda_memory<address_type>::store(announce_ + offset_in_announce, address);
          }
        }
      }
    }
  }

  DEBR_device() = delete;

  template <typename tile_type>
  DEVICE_QUALIFIER DEBR_device(const DEBR_host& host,
                               uint32_t* shared_buffer,
                               uint32_t num_active_blocks,
                               const tile_type& tile) {
    // initialize internal state
    if (tile.thread_rank() == 0) {
      limbo_bags_    = shared_buffer;
      count_per_bag_ = shared_buffer + SHARED_PTRS_COUNT;
      cur_bag_       = count_per_bag_ + NUM_BAGS;
      advance_epoch_ = cur_bag_ + 1;

      // initialize pointers to global state
      announce_             = host.d_announce_;
      current_epoch_        = host.d_current_epoch_;
      num_blocks_           = host.num_blocks_;
      num_active_bocks_     = num_active_blocks;
      external_buffer_size_ = host.external_buffer_size_;

      *advance_epoch_ = false;
      *cur_bag_       = 0;
    }
    // initialize internal storage
    initialize(tile);
  }

  __device__ ~DEBR_device() {}

  DEBR_device(const DEBR_device& DEBR_device) = delete;

  template <typename tile_type>
  DEVICE_QUALIFIER void initialize(const tile_type& tile) {
    tile.sync();
    // initialize limbo_bags
    auto stride         = tile.size();
    auto rounded_blocks = (SHARED_PTRS_COUNT + stride - 1) / stride;
    auto thread_rank    = tile.thread_rank();
    for (int i = thread_rank; i < rounded_blocks * stride; i += stride) { limbo_bags_[i] = 0; }
    // initialize count_per_bag
    if (thread_rank < NUM_BAGS) { count_per_bag_[thread_rank] = 0; }
    tile.sync();
  }
  __device__ __host__ static constexpr uint32_t compute_shmem_requirements() {
    uint32_t required_shmem = SHARED_PTRS_COUNT;  // storage of limbo bags
    required_shmem += NUM_BAGS;                   // counters per bag
    required_shmem += 1;                          // for cur_bag_
    required_shmem += 1;                          // for advance_epoch_
    return required_shmem;
  }

  static constexpr uint32_t MAX_SHMEM_KIBS = 96;                   // hardware specific
  static constexpr uint32_t PTR_SIZE_BYTES = 4;                    // implementation specific
  static constexpr uint32_t MAX_SHMEM_PTRS = MAX_SHMEM_KIBS << 8;  //
  static constexpr uint32_t NUM_BAGS       = 3;
  static_assert(SHARED_PTRS_COUNT <= MAX_SHMEM_PTRS);
  static constexpr uint32_t PTRS_PER_BAG = SHARED_PTRS_COUNT / NUM_BAGS;  // three limbo bags

  static constexpr uint32_t quiescent_bit_mask = 0x00000001;
  static constexpr uint32_t INVALID_ADDRESS    = 0xffffffff;

  /*
  // local per SM data
  */
  volatile address_type* limbo_bags_;
  volatile uint32_t* count_per_bag_;
  volatile uint32_t* cur_bag_;
  volatile uint32_t* advance_epoch_;
  /*
  // global reclaimer data
  */
  uint32_t* announce_;       // an array of num_blocks values each contain
  uint32_t* current_epoch_;  // a single number

  uint32_t num_blocks_;
  uint32_t num_active_bocks_;
  uint32_t external_buffer_size_;
};
}  // namespace smr
