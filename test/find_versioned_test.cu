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

#include <cuda_profiler_api.h>
#include <stdlib.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <cmd.hpp>
#include <cstdint>
#include <device_bump_allocator.hpp>
#include <gpu_timer.hpp>
#include <numeric>
#include <random>
#include <rkg.hpp>
#include <slab_alloc.hpp>
#include <string>
#include <unordered_set>
#include <validation.hpp>
#include <vector>

template <typename AllocatorT>
void batched_insertion_lookup(int argc, char** argv) {
  auto arguments      = std::vector<std::string>(argv, argv + argc);
  uint32_t num_keys   = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1'000'000);
  uint32_t batch_size = get_arg_value<uint32_t>(arguments, "batch-size").value_or(500000);
  int device_id       = get_arg_value<int>(arguments, "device").value_or(0);
  bool plot_dot       = get_arg_value<bool>(arguments, "plot").value_or(false);
  bool validate       = get_arg_value<bool>(arguments, "validate").value_or(true);
  bool in_place       = get_arg_value<bool>(arguments, "inplace").value_or(true);

  int device_count;
  cudaGetDeviceCount(&device_count);
  cudaDeviceProp devProp;
  if (device_id < device_count) {
    cudaSetDevice(device_id);
    cudaGetDeviceProperties(&devProp, device_id);
    std::cout << "Device[" << device_id << "]: " << devProp.name << std::endl;
  } else {
    std::cout << "No capable CUDA device found." << std::endl;
    std::terminate();
  }

  std::cout << "Inserting: " << num_keys << std::endl;
  std::cout << "Batch_size: " << batch_size << std::endl;

  uint32_t num_batches = (num_keys + batch_size - 1) / batch_size;

  static constexpr int branching_factor = 16;

  using key_type   = uint32_t;
  using value_type = uint32_t;
  using pair_type  = pair_type<key_type, value_type>;
  using node_type  = GpuBTree::node_type<key_type, value_type, branching_factor>;

  static constexpr key_type invalid_key     = std::numeric_limits<uint32_t>::max();
  static constexpr value_type invalid_value = std::numeric_limits<uint32_t>::max();

  auto to_value = [] __host__ __device__(key_type x) { return x * 10; };

  auto d_keys   = thrust::device_vector<key_type>(num_keys, invalid_key);
  auto d_values = thrust::device_vector<value_type>(num_keys, invalid_value);

  auto h_find_keys = std::vector<value_type>(num_keys, invalid_key);
  auto d_find_keys = thrust::device_vector<value_type>(num_keys, invalid_key);
  auto d_results   = thrust::device_vector<value_type>(num_keys, invalid_value);

  unsigned seed = 0;
  std::random_device rd;
  std::mt19937 rng(seed);

  auto h_keys =
      rkg::generate_keys<key_type>(num_keys * 2, rng, rkg::distribution_type::unique_random);
  rkg::prep_experiment_find_with_exist_ratio<key_type, value_type>(
      1.0, num_keys, h_keys, h_find_keys);
  h_keys.resize(num_keys);  // from num_keys * 2  => num_keys

  // copy to device
  d_keys      = h_keys;
  d_find_keys = h_find_keys;

  // assign values
  thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), to_value);
  GpuBTree::gpu_versioned_btree<key_type, value_type, branching_factor, AllocatorT> tree;

  cudaStream_t find_stream;
  cudaStream_t insertion_stream;
  cudaStreamCreate(&find_stream);
  cudaStreamCreate(&insertion_stream);

  std::vector<thrust::device_vector<value_type>> d_batch_results;
  std::vector<thrust::host_vector<value_type>> h_batch_results;
  for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
    d_batch_results.push_back(thrust::device_vector<value_type>(num_keys, invalid_value));
    h_batch_results.push_back(thrust::host_vector<value_type>(num_keys, invalid_value));
  }
  std::vector<uint32_t> timestamps(num_batches, 0);

  std::cout << "num_inserted: " << batch_size << std::endl;
  std::cout << "num_find: " << num_keys << std::endl;

  std::vector<gpu_timer*> insertion_timers;
  std::vector<gpu_timer*> find_timers;
  for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
    insertion_timers.push_back(new gpu_timer(insertion_stream));
    find_timers.push_back(new gpu_timer(find_stream));
  }

  cuda_try(cudaProfilerStart());
  for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
    auto cur_offset  = batch_size * batch_id;
    auto last_offset = std::min(cur_offset + batch_size, num_keys);
    auto cur_size    = last_offset - cur_offset;

    insertion_timers[batch_id]->start_timer();
    tree.insert(d_keys.data().get() + cur_offset,
                d_values.data().get() + cur_offset,
                cur_size,
                insertion_stream,
                in_place);
    insertion_timers[batch_id]->stop_timer();

    // wait for insertion to complete
    cuda_try(cudaStreamSynchronize(insertion_stream));

    // Take a snapshot on the find stream
    // This includes a memcpy, so it implicitly synchronize
    // i.e., it is safe to run the next insertion kernel after this call
    timestamps[batch_id] = tree.take_snapshot(find_stream);

    if (plot_dot) {
      std::string dot_name = "sc_tree";
      dot_name += std::to_string(batch_id);
      tree.plot_dot(dot_name, true);
    }

    find_timers[batch_id]->start_timer();
    tree.find(d_keys.data().get(),
              d_batch_results[batch_id].data().get(),
              num_keys,
              timestamps[batch_id],
              find_stream);
    find_timers[batch_id]->stop_timer();
  }

  cuda_try(cudaDeviceSynchronize());
  cuda_try(cudaProfilerStop());

  for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
    float insertion_seconds = insertion_timers[batch_id]->get_elapsed_s();
    float find_seconds      = find_timers[batch_id]->get_elapsed_s();

    float insertion_rate = float(batch_size) * float(1e-6) / insertion_seconds;
    float query_rate     = float(num_keys) * float(1e-6) / find_seconds;
    std::cout << "--------------  batch: " << batch_id << " --------------\n";
    std::cout << "insertion_rate = " << insertion_rate << " mop/s";
    std::cout << " (" << insertion_seconds * 1000.0f << " ms)\n";
    std::cout << "query_rate = " << query_rate << " mop/s";
    std::cout << " (" << find_seconds * 1000.0f << " ms)\n";
  }

  if (validate) {
    for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
      cuda_try(cudaMemcpyAsync(thrust::raw_pointer_cast(h_batch_results[batch_id].data()),
                               thrust::raw_pointer_cast(d_batch_results[batch_id].data()),
                               d_batch_results[batch_id].size() * sizeof(value_type),
                               cudaMemcpyDeviceToHost,
                               find_stream));

      auto cur_offset  = batch_size * batch_id;
      auto last_offset = cur_offset + batch_size;
      last_offset      = std::min(last_offset, num_keys);
      auto cur_size    = last_offset - cur_offset;
      std::cout << "Validating TS=" << timestamps[batch_id] << std::endl;
      utils::validate(
          h_keys, cur_offset + cur_size, h_find_keys, d_batch_results[batch_id], to_value);
    }
  }

  cudaStreamDestroy(insertion_stream);
  cudaStreamDestroy(find_stream);

  for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
    delete insertion_timers[batch_id];
  }
}
int main(int argc, char** argv) {
  auto arguments     = std::vector<std::string>(argv, argv + argc);
  bool use_slaballoc = get_arg_value<bool>(arguments, "slaballoc").value_or(false);

  // SlabAlloc parameters
  static constexpr uint32_t NumSuperBlocks  = 4;
  static constexpr uint32_t NumMemoryBlocks = 1024 * 8;
  static constexpr uint32_t TileSize        = 16;  // must matches the branching factor
  static constexpr uint32_t SlabSize        = 128;

  using key_type   = uint32_t;
  using value_type = uint32_t;
  using node_type  = GpuBTree::node_type<key_type, value_type, TileSize>;

  using slab_allocator_type = device_allocator::
      SlabAllocLight<node_type, NumSuperBlocks, NumMemoryBlocks, TileSize, SlabSize>;
  using bump_allocator_type = device_bump_allocator<node_type>;

  if (use_slaballoc) {
    std::cout << "Using SlabAlloc\n";
    batched_insertion_lookup<slab_allocator_type>(argc, argv);
  } else {
    std::cout << "Using Bump allocator\n";
    batched_insertion_lookup<bump_allocator_type>(argc, argv);
  }
}
