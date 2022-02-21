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
#include <stdlib.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <cmd.hpp>
#include <cstdint>
#include <gpu_timer.hpp>
#include <numeric>
#include <random>
#include <rkg.hpp>
#include <string>
#include <unordered_set>
#include <validation.hpp>
#include <vector>

#include <device_bump_allocator.hpp>
#include <slab_alloc.hpp>

namespace cg = cooperative_groups;

template <typename Allocator, int B>
__global__ void kernel(Allocator allocator) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block     = cg::this_thread_block();
  auto tile      = cg::tiled_partition<B>(block);
  auto tile_id   = thread_id / B;

  using allocator_type = device_allocator_context<Allocator>;
  allocator_type device_allocator{allocator, tile};
  const size_t num_to_allocate = 33554432;
  uint32_t ptrs[num_to_allocate];
  if (tile_id == 1) {
    for (size_t i = 0; i < num_to_allocate; i++) {
      ptrs[i] = device_allocator.allocate(allocator, 1, tile);
    }
    for (size_t i = 0; i < num_to_allocate; i++) {
      if (tile.thread_rank() == 0) { device_allocator.deallocate(allocator, ptrs[i], 0); }
    }
  }
}

template <int B = 16>
struct node {
  uint32_t pairs[B * 2];
};
int main(int argc, char** argv) {
  //{ SlabAllocLight<node_type, 13, 8, 32> a; }  // 8 GiBs
  //{ SlabAllocLight<node_type, 12, 8, 32> a; }  // 4 GiBs
  (void)argc;
  (void)argv;
  static constexpr int B = 16;
  using node_type        = node<B>;

  using slab_allocator_type = device_allocator::SlabAllocLight<node_type, 8, 1024 * 8, 16, 128>;

  slab_allocator_type allocator;

  allocator.get_allocated_count();

  kernel<slab_allocator_type, B><<<1, 64>>>(allocator);
  cuda_try(cudaDeviceSynchronize());

  allocator.get_allocated_count();
}
