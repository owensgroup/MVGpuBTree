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
#include <type_traits>

enum class cuda_memory_order {
  memory_order_weak,
  memory_order_relaxed,
  memory_order_consume,
  memory_order_acquire,
  memory_order_release,
  memory_order_acq_rel,
  memory_order_seq_cst

};
template <typename T>
struct cuda_memory_64 {
  static_assert(sizeof(T) == sizeof(uint64_t));
  using unsigned_type = uint64_t;

  __device__ static inline T load(T* ptr,
                                  cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    unsigned_type old;
    switch (order) {
      case cuda_memory_order::memory_order_weak:
        asm volatile("ld.weak.global.b64 %0,[%1];"
                     : "=l"(old)
                     : "l"(reinterpret_cast<unsigned_type*>(ptr))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_relaxed:
        asm volatile("ld.relaxed.gpu.b64 %0,[%1];"
                     : "=l"(old)
                     : "l"(reinterpret_cast<unsigned_type*>(ptr))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_consume:
        asm volatile("ld.acquire.gpu.b64 %0,[%1];"
                     : "=l"(old)
                     : "l"(reinterpret_cast<unsigned_type*>(ptr))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_acquire:
        asm volatile("ld.acquire.gpu.b64 %0,[%1];"
                     : "=l"(old)
                     : "l"(reinterpret_cast<unsigned_type*>(ptr))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_release: cuda_assert(false); break;
      case cuda_memory_order::memory_order_acq_rel: cuda_assert(false); break;
      case cuda_memory_order::memory_order_seq_cst:
        asm volatile("fence.sc.gpu;" ::: "memory");
        asm volatile("ld.acquire.gpu.b64 %0,[%1];"
                     : "=l"(old)
                     : "l"(reinterpret_cast<unsigned_type*>(ptr))
                     : "memory");
        break;
      default: cuda_assert(false); break;
    }

    return *reinterpret_cast<T*>(&old);
  }
  __device__ static inline void
  store(T* ptr, T value, cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    switch (order) {
      case cuda_memory_order::memory_order_weak:
        asm volatile("st.weak.global.b64 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                     "l"(*reinterpret_cast<unsigned_type*>(&value))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_relaxed:
        asm volatile("st.relaxed.gpu.b64 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                     "l"(*reinterpret_cast<unsigned_type*>(&value))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_consume: cuda_assert(false); break;
      case cuda_memory_order::memory_order_acquire: cuda_assert(false); break;
      case cuda_memory_order::memory_order_release:
        asm volatile("st.release.gpu.b64 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                     "l"(*reinterpret_cast<unsigned_type*>(&value))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_acq_rel: cuda_assert(false); break;
      case cuda_memory_order::memory_order_seq_cst:
        asm volatile("fence.sc.gpu;" ::: "memory");
        asm volatile("st.relaxed.gpu.b64 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                     "l"(*reinterpret_cast<unsigned_type*>(&value))
                     : "memory");
        break;
      default: cuda_assert(false); break;
    }
  }
};

template <typename T>
struct cuda_memory_32 {
  static_assert(sizeof(T) == sizeof(uint32_t));
  using unsigned_type = uint32_t;

  __device__ static inline T load(T* ptr,
                                  cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    unsigned_type old;
    switch (order) {
      case cuda_memory_order::memory_order_weak:
        asm volatile("ld.weak.global.b32 %0,[%1];"
                     : "=r"(old)
                     : "l"(reinterpret_cast<unsigned_type*>(ptr))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_relaxed:
        asm volatile("ld.relaxed.gpu.b32 %0,[%1];"
                     : "=r"(old)
                     : "l"(reinterpret_cast<unsigned_type*>(ptr))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_consume:
        asm volatile("ld.acquire.gpu.b32 %0,[%1];"
                     : "=r"(old)
                     : "l"(reinterpret_cast<unsigned_type*>(ptr))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_acquire:
        asm volatile("ld.acquire.gpu.b32 %0,[%1];"
                     : "=r"(old)
                     : "l"(reinterpret_cast<unsigned_type*>(ptr))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_release: cuda_assert(false); break;
      case cuda_memory_order::memory_order_acq_rel: cuda_assert(false); break;
      case cuda_memory_order::memory_order_seq_cst:
        asm volatile("fence.sc.gpu;" ::: "memory");
        asm volatile("ld.acquire.gpu.b32 %0,[%1];"
                     : "=r"(old)
                     : "l"(reinterpret_cast<unsigned_type*>(ptr))
                     : "memory");
        break;
      default: cuda_assert(false); break;
    }

    return *reinterpret_cast<T*>(&old);
  }
  __device__ static inline void
  store(T* ptr, T value, cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    switch (order) {
      case cuda_memory_order::memory_order_weak:
        asm volatile("st.weak.global.b32 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                     "r"(*reinterpret_cast<unsigned_type*>(&value))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_relaxed:
        asm volatile("st.relaxed.gpu.b32 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                     "r"(*reinterpret_cast<unsigned_type*>(&value))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_consume: cuda_assert(false); break;
      case cuda_memory_order::memory_order_acquire: cuda_assert(false); break;
      case cuda_memory_order::memory_order_release:
        asm volatile("st.release.gpu.b32 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                     "r"(*reinterpret_cast<unsigned_type*>(&value))
                     : "memory");
        break;
      case cuda_memory_order::memory_order_acq_rel: cuda_assert(false); break;
      case cuda_memory_order::memory_order_seq_cst:
        asm volatile("fence.sc.gpu;" ::: "memory");
        asm volatile("st.relaxed.gpu.b32 [%0], %1;" ::"l"(reinterpret_cast<unsigned_type*>(ptr)),
                     "r"(*reinterpret_cast<unsigned_type*>(&value))
                     : "memory");
        break;
      default: cuda_assert(false); break;
    }
  }
};

template <typename T = uint32_t>
struct cuda_memory
    : public std::conditional<sizeof(T) == 4, cuda_memory_32<T>, cuda_memory_64<T>>::type {
  static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

  __device__ static inline void atomic_thread_fence() {
    asm volatile("fence.sc.gpu;" ::: "memory");
  }
};
