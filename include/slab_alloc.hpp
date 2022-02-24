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
#include <thrust/device_vector.h>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <macros.hpp>
#include <memory>
#include <type_traits>

//#define DEBUG_GLOBAL_ALLOCATE
//#define DEBUG_GLOBAL_DEALLOCATE
// #define DEBUG_GLOBAL_ALLOCATE_DEALLOCATE

//#define DEBUG
// #define COLLECT_ALLOCATOR_STATS
//#define DEBUG_FULL_BLOCKS_REHASH
#ifdef DEBUG
#define debug_print_tile(tile, fmt, ...)        \
  if (tile.thread_rank() == 0) {                \
    do { printf(fmt, __VA_ARGS__); } while (0); \
  }
#else
#define debug_print_tile(fmt, ...)
#endif

template <typename T>
constexpr bool is_pow2(T x) {
  return (x != 0) && ((x & (x - 1)) == 0);
}

constexpr uint32_t clz(uint32_t x) {
  const char debruijn32[32] = {0, 31, 9, 30, 3, 8,  13, 29, 2,  5,  7,  21, 12, 24, 28, 19,
                               1, 10, 4, 14, 6, 22, 25, 20, 11, 15, 23, 26, 16, 27, 17, 18};
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return debruijn32[x * 0x076be629 >> 27];
}

constexpr uint32_t bit_width(uint32_t x) {
  using T = uint32_t;
  return static_cast<T>(std::numeric_limits<T>::digits - clz(x));
}
namespace device_allocator {

using pointer                  = uint32_t;
constexpr auto invalid_address = std::numeric_limits<pointer>::max();

namespace detail {

template <typename T>
DEVICE_QUALIFIER int cuda_ffs(const T& x) {
  return __ffsll(x);
}
DEVICE_QUALIFIER int cuda_ffs(const uint32_t& x) { return __ffs(x); }

template <class Allocator>
struct GenericAllocator {
  static inline constexpr void deallocate(pointer ptr, std::byte* base_ptr) {
    uint32_t index         = get_level_index(ptr);
    std::byte* level_start = get_level_start(index, base_ptr);
    Allocator::sublevel_type::deallocate(ptr, level_start);
  }
  static inline constexpr bool is_allocated(pointer ptr, std::byte* base_ptr) {
    uint32_t index         = get_level_index(ptr);
    std::byte* level_start = get_level_start(index, base_ptr);
    return Allocator::sublevel_type::is_allocated(ptr, level_start);
  }
  static inline constexpr std::byte* address(pointer ptr, std::byte* base_ptr) {
    uint32_t index         = get_level_index(ptr);
    std::byte* level_start = get_level_start(index, base_ptr);
    return Allocator::sublevel_type::address(ptr, level_start);
  }
  static inline constexpr uint32_t get_level_index(pointer ptr) {
    pointer level_index = ptr & Allocator::level_address_mask;
    level_index         = level_index >> Allocator::required_sublevels_bits_count;
    return level_index;
  }
  static inline constexpr std::byte* get_level_start(uint32_t index, std::byte* base_ptr) {
    auto one_level_size    = (Allocator::level_size / Allocator::level_count);
    uint64_t level_offset  = one_level_size * index;
    std::byte* level_start = base_ptr + level_offset;
    return level_start;
  }
};

template <uint32_t TileSize, uint32_t SlabSize>
struct Block {
  static_assert(is_pow2(SlabSize));

  using bitmap_type = std::conditional_t<TileSize == 32, unsigned int, unsigned long long int>;
  using level_type  = Block<TileSize, SlabSize>;

  static constexpr uint32_t bitmap_size  = sizeof(bitmap_type);
  static constexpr uint32_t bitmaps_size = sizeof(bitmap_type) * TileSize;

  static constexpr uint32_t level_count = bitmaps_size << 3;  // 8 bits per byte
  static constexpr uint64_t level_size  = (level_count + 1) * SlabSize;

  static constexpr uint32_t required_bits_count           = bit_width(level_count - 1);
  static constexpr uint32_t required_sublevels_bits_count = 0;

  Block() = default;
  template <typename tile_type>
  DEVICE_QUALIFIER Block(std::byte* level_start, const tile_type& tile)
      : level_start_(level_start) {
    bitmap_start_ = reinterpret_cast<bitmap_type*>(level_start);
    bitmap_       = bitmap_start_[tile.thread_rank()];
  }

  template <typename tile_type>
  DEVICE_QUALIFIER pointer allocate(tile_type& tile) {
    uint32_t allocated_result = invalid_address;
    while (allocated_result == invalid_address) {
      auto empty_lane = cuda_ffs(~bitmap_) - 1;
      auto free_lane  = tile.ballot(empty_lane >= 0);
      if (free_lane != 0) {
        bitmap_type read_bitmap = bitmap_;
        uint32_t src_lane       = cuda_ffs(free_lane) - 1;
        if (src_lane == tile.thread_rank()) {
          bitmap_type desired_unit  = static_cast<bitmap_type>(1) << empty_lane;
          bitmap_type expected_bmap = bitmap_;
          bitmap_type desired_bmap  = bitmap_ | desired_unit;
          read_bitmap = atomicCAS(bitmap_start_ + tile.thread_rank(), expected_bmap, desired_bmap);
          if (read_bitmap == expected_bmap) {
            bitmap_          = desired_bmap;  // update register
            allocated_result = empty_lane + src_lane * sizeof(bitmap_type) * 8;
#ifdef DEBUG_GLOBAL_ALLOCATE_DEALLOCATE
            printf("allocated at %p @[%i]\n", bitmap_start_ + tile.thread_rank(), empty_lane);
#endif
          } else {
            bitmap_ = read_bitmap;
          }
        }
        allocated_result = tile.shfl(allocated_result, src_lane);
      } else {
        return invalid_address;  // No space
      }
    }
    debug_print_tile(tile,
                     "tile = %u: allocated memory unit %u (0x%08x)\n",
                     (threadIdx.x + blockIdx.x * blockDim.x) / TileSize,
                     allocated_result,
                     allocated_result);
    return allocated_result;
  }

  static inline constexpr uint32_t get_bitmap_thread_rank(const uint32_t& index) {
    return index / (sizeof(bitmap_type) * 8);
  }
  static inline constexpr uint32_t get_bitmap_unit_rank(const uint32_t& index) {
    return index % (sizeof(bitmap_type) * 8);
  }

  static constexpr uint32_t compute_required_bits() { return required_bits_count; }

  static inline constexpr std::byte* address(pointer ptr, std::byte* level_start) {
    uint32_t index             = GenericAllocator<level_type>::get_level_index(ptr);
    std::byte* ptr_address     = level_start + bitmaps_size;
    std::byte* ptr_plus_header = &ptr_address[index * SlabSize];
    return ptr_plus_header;
  }

  static inline constexpr void deallocate(pointer ptr, std::byte* level_start) {
    // note that this free call can free from a non-resident memory block
    // hence, it requires the memory block pointer
    uint32_t index               = GenericAllocator<level_type>::get_level_index(ptr);
    auto bitmap_rank             = get_bitmap_thread_rank(index);
    bitmap_type* bmap_address    = reinterpret_cast<bitmap_type*>(level_start) + bitmap_rank;
    bitmap_type bitmap_unit_rank = get_bitmap_unit_rank(index);
    bitmap_type mask             = static_cast<bitmap_type>(1) << bitmap_unit_rank;
    auto old                     = atomicAnd(bmap_address, ~mask);
#ifdef DEBUG_GLOBAL_ALLOCATE_DEALLOCATE
    printf("deallocated at %p [%i]\n", bmap_address, bitmap_unit_rank);
#endif
#ifdef DEBUG
    // check for doublefree
    assert((old & mask) != 0);
    printf("Dellocated at %i [%u, %llu] -> bmap (0x%llx -> 0x%llx) \n",
           index,
           bitmap_rank,
           bitmap_unit_rank,
           old,
           old & (~mask));
#endif
  }

  static inline constexpr bool is_allocated(pointer ptr, std::byte* level_start) {
    uint32_t index               = GenericAllocator<level_type>::get_level_index(ptr);
    auto bitmap_rank             = get_bitmap_thread_rank(index);
    bitmap_type* bmap_address    = reinterpret_cast<bitmap_type*>(level_start) + bitmap_rank;
    bitmap_type bitmap_unit_rank = get_bitmap_unit_rank(index);
    bitmap_type mask             = static_cast<bitmap_type>(1) << bitmap_unit_rank;
    auto old                     = *bmap_address;
    return ((old & mask) != 0);
  }

  static_assert(level_count == 1024);
  static constexpr uint32_t level_address_mask = level_count - 1;

 private:
  uint32_t index_;
  std::byte* level_start_;
  bitmap_type* bitmap_start_;
  bitmap_type bitmap_;
};

template <uint32_t Count, uint32_t TileSize, uint32_t SlabSize>
struct MemoryBlock {
  static_assert(is_pow2(Count));
  static constexpr uint32_t level_count = Count;
  using level_type                      = MemoryBlock<Count, TileSize, SlabSize>;
  using sublevel_type                   = Block<TileSize, SlabSize>;
  static constexpr uint64_t level_size  = sublevel_type::level_size * Count;

  MemoryBlock() = default;
  template <typename tile_type>
  DEVICE_QUALIFIER MemoryBlock(uint32_t index, std::byte* level_start, tile_type& tile)
      : index_(index), level_start_(level_start) {
    auto block_start = GenericAllocator<level_type>::get_level_start(index, level_start);
    sublevel_        = sublevel_type(block_start, tile);
  }
  template <typename tile_type>
  DEVICE_QUALIFIER pointer allocate(std::size_t count, tile_type& tile) {
    pointer sub_alloc = sublevel_.allocate(tile);

    if (sub_alloc == invalid_address) { return invalid_address; }
    pointer level_address = index_ << required_sublevels_bits_count;
    return level_address | sub_alloc;
  }
  static inline constexpr std::byte* address(pointer ptr, std::byte* level_start) {
    return GenericAllocator<level_type>::address(ptr, level_start);
  }
  static inline constexpr void deallocate(pointer ptr, std::byte* level_start) {
    GenericAllocator<level_type>::deallocate(ptr, level_start);
  }
  static inline constexpr bool is_allocated(pointer ptr, std::byte* level_start) {
    return GenericAllocator<level_type>::is_allocated(ptr, level_start);
  }
  static constexpr uint32_t compute_required_bits() {
    return required_bits_count + required_sublevels_bits_count;
  }

  static constexpr uint32_t required_bits_count           = bit_width(level_count - 1);
  static constexpr uint32_t required_sublevels_bits_count = sublevel_type::compute_required_bits();
  static constexpr uint32_t level_address_mask = (level_count - 1) << required_sublevels_bits_count;

 private:
  uint32_t index_;
  sublevel_type sublevel_;
  std::byte* level_start_;
};

template <uint32_t Count, uint32_t NumMemoryBlocks, uint32_t TileSize, uint32_t SlabSize>
struct SuperBlock {
  static_assert(is_pow2(Count));
  static constexpr uint32_t level_count = Count;
  using level_type                      = SuperBlock<Count, NumMemoryBlocks, TileSize, SlabSize>;
  using sublevel_type                   = MemoryBlock<NumMemoryBlocks, TileSize, SlabSize>;
  static constexpr uint64_t level_size  = sublevel_type::level_size * Count;

  SuperBlock() = default;
  template <typename tile_type>
  DEVICE_QUALIFIER SuperBlock(std::byte* level_start, tile_type& tile) : level_start_(level_start) {
    // distribute the superblocks across the machine
    auto thread_id   = threadIdx.x + blockIdx.x * blockDim.x;
    auto tile_id     = thread_id / TileSize;
    index_           = tile_id % level_count;
    num_attempts_    = 0;
    auto block_start = GenericAllocator<level_type>::get_level_start(index_, level_start);
    sub_index_ = uint32_t((uint64_t(hash_coef_) * uint64_t(tile_id)) % uint64_t(NumMemoryBlocks));

    sublevel_ = sublevel_type(sub_index_, block_start, tile);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void rehash(tile_type& tile) {
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    auto tile_id   = thread_id / TileSize;

    // linearly probe the blocks
    num_attempts_++;
    sub_index_++;
    if (sub_index_ == NumMemoryBlocks) { sub_index_ = 0; }
    auto first_sub_index =
        uint32_t((uint64_t(hash_coef_) * uint64_t(tile_id)) % uint64_t(NumMemoryBlocks));
    if (sub_index_ == first_sub_index) {
      index_++;
      index_ = index_ % level_count;

      // auto first_index = tile_id % level_count;
      // if index_ reaches first block (i.e., tile_id % level_count) then
      //  the allocator is completely full
      // if (index_ == first_index) { assert(0); }
    }
    auto block_start = GenericAllocator<level_type>::get_level_start(index_, level_start_);

#ifdef DEBUG_FULL_BLOCKS_REHASH
    if (threadIdx.x == 0) {
      printf("block %i: num_attempts = %i, index_ = %i, sub_index = %i\n",
             blockIdx.x,
             num_attempts_,
             index_,
             sub_index_);
    }
#endif
    sublevel_ = sublevel_type{sub_index_, block_start, tile};
  }
  template <typename tile_type>
  __device__ pointer allocate(std::size_t count, tile_type& tile) {
    pointer sub_address = sublevel_.allocate(count, tile);
    while (sub_address == invalid_address) {
      rehash(tile);
      sub_address = sublevel_.allocate(count, tile);
    }

    pointer level_address = index_ << required_sublevels_bits_count;
    return level_address | sub_address;
  }
  static constexpr std::byte* address(pointer ptr, std::byte* level_start) {
    return GenericAllocator<level_type>::address(ptr, level_start);
  }
  static constexpr void deallocate(pointer ptr, std::byte* level_start) {
    GenericAllocator<level_type>::deallocate(ptr, level_start);
  }
  static constexpr bool is_allocated(pointer ptr, std::byte* level_start) {
    return GenericAllocator<level_type>::is_allocated(ptr, level_start);
  }
  static constexpr uint32_t compute_required_bits() {
    return required_bits_count + required_sublevels_bits_count;
  }

  static constexpr uint32_t required_bits_count           = bit_width(level_count - 1);
  static constexpr uint32_t required_sublevels_bits_count = sublevel_type::compute_required_bits();
  static constexpr uint32_t level_address_mask = (level_count - 1) << required_sublevels_bits_count;

 private:
  uint32_t index_;
  uint32_t sub_index_;
  sublevel_type sublevel_;
  std::byte* level_start_;
  uint32_t num_attempts_;
  static constexpr uint32_t hash_coef_ = 0x5904;
};
}  // namespace detail

template <class T,
          uint32_t NumSuperBlocks  = 2,
          uint32_t NumMemoryBlocks = 10,
          unsigned TileSize        = 32,
          uint32_t SlabSize        = 128>
struct SlabAllocator {
  SlabAllocator() = default;
  template <typename tile_type>
  DEVICE_QUALIFIER SlabAllocator(void* ptr, tile_type& tile) {
    pool_  = reinterpret_cast<std::byte*>(ptr);
    block_ = block_type(pool_, tile);
  }
  template <typename tile_type>
  DEVICE_QUALIFIER pointer allocate(std::size_t size, tile_type& tile) {
    auto allocation = block_.allocate(size, tile);
#ifdef DEBUG_GLOBAL_ALLOCATE_DEALLOCATE
    bool real_ptr = is_allocated(allocation);
    assert(real_ptr);
#endif
    return allocation;
  }
  DEVICE_QUALIFIER void deallocate(pointer ptr, std::size_t size) {
#ifdef DEBUG_GLOBAL_ALLOCATE_DEALLOCATE
    bool real_ptr = is_allocated(ptr);
    assert(real_ptr);
#endif
    block_type::deallocate(ptr, pool_);
  }
  DEVICE_QUALIFIER bool is_allocated(pointer ptr) { return block_type::is_allocated(ptr, pool_); }

  DEVICE_QUALIFIER T* address(pointer ptr) const {
    return reinterpret_cast<T*>(block_type::address(ptr, pool_));
  }

  static constexpr uint32_t compute_required_bits() { return block_type::compute_required_bits(); }

  static constexpr uint64_t get_required_size() { return block_type::level_size; }
  std::byte* pool_;

 private:
  using block_type =
      device_allocator::detail::SuperBlock<NumSuperBlocks, NumMemoryBlocks, TileSize, SlabSize>;
  block_type block_;
};

template <class T,
          uint32_t NumSuperBlocks  = 2,
          uint32_t NumMemoryBlocks = 10,
          unsigned TileSize        = 32,
          uint32_t SlabSize        = 128>
struct SlabAllocLight {
  using DeviceAllocator = SlabAllocator<T, NumSuperBlocks, NumMemoryBlocks, TileSize, SlabSize>;
  using value_type      = T;
  using pointer_type    = uint32_t;

  void preallocate() {
    auto bytes_count = DeviceAllocator::get_required_size();
#ifdef COLLECT_ALLOCATOR_STATS
    bytes_count += 8;  // last two numbers are #currently allocated #total allocated
#endif
    cudaMalloc(&pool, bytes_count);
    cudaMemset(pool, 0x00, bytes_count);
  }

  SlabAllocLight() {
    preallocate();
    is_copy_ = false;
  }
  SlabAllocLight(const SlabAllocLight& other) {
    pool     = other.pool;
    is_copy_ = true;
  }

  void copy_buffer(T* buffer, uint32_t bytes_count) const {
    (void)buffer;
    (void)bytes_count;
  }
  SlabAllocLight& operator=(const SlabAllocLight& other) {
    pool     = other.pool;
    is_copy_ = true;
    return *this;
  }

  ~SlabAllocLight() {
    if (!is_copy_) { cuda_try(cudaFree(pool)); }
  }

  uint32_t get_allocated_count() const { return 0; }
  void* pool;
  bool is_copy_;
};

}  // namespace device_allocator

template <class T,
          uint32_t NumSuperBlocks,
          uint32_t NumMemoryBlocks,
          unsigned TileSize,
          uint32_t SlabSize>
struct device_allocator_context<
    device_allocator::SlabAllocLight<T, NumSuperBlocks, NumMemoryBlocks, TileSize, SlabSize>> {
  using allocator_type =
      device_allocator::SlabAllocLight<T, NumSuperBlocks, NumMemoryBlocks, TileSize, SlabSize>;
  using size_type = std::size_t;

  // called at the beginning of the kernel:
  template <typename tile_type>
  DEVICE_QUALIFIER device_allocator_context(allocator_type& alloc, const tile_type& tile)
      : device_allocator_{alloc.pool, tile} {}

  template <typename tile_type>
  DEVICE_QUALIFIER device_allocator::pointer allocate(allocator_type& a,
                                                      std::size_t n,
                                                      const tile_type& tile) {
    auto allocated_result = device_allocator_.allocate(n, tile);
#ifdef DEBUG_GLOBAL_ALLOCATE
    if (tile.thread_rank() == 0) {
      printf("tile = %u: allocated memory unit %u (0x%08x)\n",
             (threadIdx.x + blockIdx.x * blockDim.x) / TileSize,
             allocated_result,
             allocated_result);
    }
#endif
#ifdef COLLECT_ALLOCATOR_STATS
    if (tile.thread_rank() == 0) {
      auto bytes_count = allocator_type::DeviceAllocator::get_required_size();
      auto counter_ptr = reinterpret_cast<int*>(device_allocator_.pool_ + bytes_count);
      atomicAdd(counter_ptr, 1);
      atomicAdd(counter_ptr + 1, 1);
    }
#endif
    return allocated_result;
  }
  DEVICE_QUALIFIER void deallocate(allocator_type& a,
                                   device_allocator::pointer ptr,
                                   std::size_t n) {
#ifdef DEBUG_GLOBAL_DEALLOCATE
    {
      printf("tid = %u: deallocated memory unit %u (0x%08x)\n",
             (threadIdx.x + blockIdx.x * blockDim.x),
             ptr,
             ptr);
    }
#endif
    device_allocator_.deallocate(ptr, n);
#ifdef COLLECT_ALLOCATOR_STATS
    auto bytes_count = allocator_type::DeviceAllocator::get_required_size();
    auto counter_ptr = reinterpret_cast<int*>(device_allocator_.pool_ + bytes_count);
    atomicAdd(counter_ptr, -1);
#endif
  }

  DEVICE_QUALIFIER bool is_allocated(device_allocator::pointer ptr) {
    return device_allocator_.is_allocated(ptr);
  }

  template <typename HostAllocator>
  DEVICE_QUALIFIER T* address(HostAllocator& a, device_allocator::pointer ptr) const {
    return device_allocator_.address(ptr);
  }
  DEVICE_QUALIFIER unsigned int get_allocated_count() {
    auto bytes_count = allocator_type::DeviceAllocator::get_required_size();
    auto ptr = reinterpret_cast<volatile unsigned int*>(device_allocator_.pool_ + bytes_count);
    return *ptr;
  }
  DEVICE_QUALIFIER unsigned int get_total_allocated_count() {
    auto bytes_count = allocator_type::DeviceAllocator::get_required_size();
    auto ptr = reinterpret_cast<volatile unsigned int*>(device_allocator_.pool_ + bytes_count);
    return *(ptr + 1);
  }

  DEVICE_QUALIFIER unsigned int get_max_allocated_count() {
    auto bytes_count = allocator_type::DeviceAllocator::get_required_size();
    return bytes_count / 128;
  }

  using device_allocator_type =
      device_allocator::SlabAllocator<T, NumSuperBlocks, NumMemoryBlocks, TileSize, SlabSize>;

  device_allocator_type device_allocator_;
};
