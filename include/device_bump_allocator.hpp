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
#include <cstdint>

// 67108864 is 8 Gibs when sizeof(T) = 128
template <class T, std::size_t MaxTCount = 67108864>
struct device_bump_allocator {
  using size_type       = uint32_t;
  using difference_type = uint32_t;

  using value_type   = T;
  using pointer_type = uint32_t;

  device_bump_allocator() {
    d_buffer_     = cuda_allocator<T>().allocate(max_size_);
    d_slab_count_ = cuda_allocator<uint32_t>().allocate(1);
    cuda_try(cudaMemset(d_slab_count_, 0x00, sizeof(uint32_t)));

    // construct shared pointers
    buffer_     = std::shared_ptr<T>(d_buffer_, cuda_deleter<T>());
    slab_count_ = std::shared_ptr<size_type>(d_slab_count_, cuda_deleter<uint32_t>());
  }
  template <class U>
  HOST_DEVICE_QUALIFIER constexpr device_bump_allocator(
      const device_bump_allocator<U>& other) noexcept
      : d_buffer_(other.d_buffer_)
      , buffer_(other.buffer_)
      , d_slab_count_(other.d_slab_count_)
      , slab_count_(other.slab_count_) {}
  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type allocate(std::size_t n, const tile_type& tile) {
    static constexpr int elected_lane = 0;
    pointer_type new_slab_index       = 0;
    if (tile.thread_rank() == elected_lane) {
      new_slab_index = atomicAdd(d_slab_count_, n);
      cuda_assert(new_slab_index != max_size_);
    }
    return tile.shfl(new_slab_index, elected_lane);
  }
  DEVICE_QUALIFIER void deallocate(pointer_type p, std::size_t n) noexcept {}

  HOST_DEVICE_QUALIFIER value_type* address(pointer_type ptr) const {
#ifdef __CUDA_ARCH__
    cuda_assert(ptr < max_size_);

#else
    assert(ptr < max_size_);
#endif
    return d_buffer_ + ptr;
  }

  uint32_t get_allocated_count() const {
    size_type count = 0;
    cuda_try(cudaMemcpy(&count, d_slab_count_, sizeof(size_type), cudaMemcpyDeviceToHost));
    return count;
  }
  __device__ void set_allocated_count(size_type count) { *d_slab_count_ = count; }
  uint32_t get_total_allocated_count() const { return get_allocated_count(); }

  void copy_buffer(T* buffer, std::size_t bytes_count) const {
    cuda_try(cudaMemcpy(buffer, d_buffer_, bytes_count, cudaMemcpyDeviceToHost));
  }

  HOST_DEVICE_QUALIFIER T* get_raw_buffer() { return d_buffer_; }

 private:
  T* d_buffer_;
  std::shared_ptr<T> buffer_;
  uint32_t* d_slab_count_;
  std::shared_ptr<uint32_t> slab_count_;
  static constexpr uint64_t max_size_ = MaxTCount;
};

template <class T, std::size_t MaxTCount>
struct device_allocator_context<device_bump_allocator<T, MaxTCount>> {
  using allocator_type = device_bump_allocator<T, MaxTCount>;
  using value_type     = typename allocator_type::value_type;
  using pointer        = typename allocator_type::pointer_type;
  using size_type      = typename allocator_type::size_type;

  template <typename tile_type>
  HOST_DEVICE_QUALIFIER device_allocator_context(allocator_type& alloc, const tile_type& tile) {}

  template <typename tile_type>
  HOST_DEVICE_QUALIFIER pointer allocate(allocator_type& a, size_type n, const tile_type& tile) {
    return a.allocate(n, tile);
  }
  HOST_DEVICE_QUALIFIER void deallocate(allocator_type& a, pointer p, size_type n) {
    a.deallocate(p, n);
  }
  HOST_DEVICE_QUALIFIER bool is_allocated(pointer) { return true; }
  HOST_DEVICE_QUALIFIER value_type* address(allocator_type& a, pointer ptr) const {
    return a.address(ptr);
  }
};
