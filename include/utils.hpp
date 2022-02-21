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

namespace utils {
namespace bits {
// Bit Field Extract.
DEVICE_QUALIFIER int bfe(uint32_t src, int num_bits) {
  unsigned mask;
  asm("bfe.u32 %0, %1, 0, %2;" : "=r"(mask) : "r"(src), "r"(num_bits));
  return mask;
}

// Find most significant non - sign bit.
// bfind(0) = -1, bfind(1) = 0
DEVICE_QUALIFIER int bfind(uint32_t src) {
  int msb;
  asm("bfind.u32 %0, %1;" : "=r"(msb) : "r"(src));
  return msb;
}
DEVICE_QUALIFIER int bfind(uint64_t src) {
  int msb;
  asm("bfind.u64 %0, %1;" : "=r"(msb) : "l"(src));
  return msb;
}
};  // namespace bits

struct device_memory_usage_results {
  std::size_t used_bytes;
  std::size_t total_bytes;
};
device_memory_usage_results compute_device_memory_usage() {
  std::size_t total_bytes;
  std::size_t free_bytes;
  cuda_try(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::size_t used_bytes = total_bytes - free_bytes;
  return {used_bytes, free_bytes};
}
void set_cuda_buffer_size(const std::size_t new_size, const cudaLimit limit) {
  cuda_try(cudaDeviceSetLimit(limit, new_size));
}

std::size_t get_cuda_buffer_size(const cudaLimit limit) {
  std::size_t cur_size;
  cuda_try(cudaDeviceGetLimit(&cur_size, limit));
  return cur_size;
}

};  // namespace utils

namespace GpuBTree {
template <typename tile_type, typename pair_type>
DEVICE_QUALIFIER void print_hex(const tile_type& tile, const pair_type& pair) {
  printf("node: rank: %i, pair{%#010x, %#010x}\n", tile, pair.first, pair.second);
}

};  // namespace GpuBTree
