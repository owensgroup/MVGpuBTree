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
#include <device_bump_allocator.hpp>
#include <gpu_timer.hpp>
#include <iterator>
#include <numeric>
#include <random>
#include <rkg.hpp>
#include <slab_alloc.hpp>
#include <string>
#include <unordered_set>
#include <vector>

template <typename key_type,
          typename value_type,
          typename size_type,
          uint32_t branching_factor = 16>
void build(size_type num_keys,
           size_type num_ranges,
           float exist_ratio,
           uint32_t average_range_length,
           bool plot_tree,
           bool validate,
           bool plot_range) {
  // roundup
  num_keys = (num_keys + branching_factor - 1) / branching_factor;
  num_keys = num_keys * branching_factor;

  num_ranges = (num_ranges + branching_factor - 1) / branching_factor;
  num_ranges = num_ranges * branching_factor;

  using pair_type                           = pair_type<key_type, value_type>;
  static constexpr key_type invalid_key     = std::numeric_limits<uint32_t>::max();
  static constexpr value_type invalid_value = std::numeric_limits<uint32_t>::max();

  auto d_keys   = thrust::device_vector<key_type>(num_keys, invalid_key);
  auto d_values = thrust::device_vector<value_type>(num_keys, invalid_value);

  auto h_find_keys_lower = std::vector<key_type>(num_ranges, invalid_key);
  auto d_find_keys_lower = thrust::device_vector<key_type>(num_ranges, invalid_key);
  auto d_find_keys_upper = thrust::device_vector<key_type>(num_ranges, invalid_key);
  auto d_range_results   = thrust::device_vector<pair_type>(num_ranges * average_range_length);

  unsigned seed = 0;
  std::random_device rd;
  std::mt19937 rng(seed);

  // Generate a dataset
  auto h_keys =
      rkg::generate_keys<key_type>(num_keys * 2, rng, rkg::distribution_type::unique_random);
  rkg::prep_experiment_find_with_exist_ratio<key_type, value_type>(
      exist_ratio, num_ranges, h_keys, h_find_keys_lower);
  h_keys.resize(num_keys);  // from num_keys * 2  => num_keys

  // copy to device
  d_keys            = h_keys;
  d_find_keys_lower = h_find_keys_lower;

  // copy to device
  d_keys = h_keys;

  // assign values
  auto to_value = [] __host__ __device__(key_type x) { return x * 10; };
  thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), to_value);

  // assign upper query bound
  auto to_upper_bound = [average_range_length] __host__ __device__(key_type x) {
    // printf("%i\n", average_range_length);
    return x + average_range_length;
  };
  thrust::transform(thrust::device,
                    d_find_keys_lower.begin(),
                    d_find_keys_lower.end(),
                    d_find_keys_upper.begin(),
                    to_upper_bound);

  // Tree
  using node_type           = GpuBTree::node_type<key_type, value_type, branching_factor>;
  using bump_allocator_type = device_bump_allocator<node_type>;
  std::cout << "branching_factor: " << branching_factor << std::endl;
  std::cout << "sizeof(node_type): " << sizeof(node_type) << std::endl;

  static constexpr uint32_t NumSuperBlocks  = 4;
  static constexpr uint32_t NumMemoryBlocks = 1024 * 8;
  static constexpr uint32_t TileSize        = 16;
  static constexpr uint32_t SlabSize        = 128;

  using slab_allocator_type = device_allocator::
      SlabAllocLight<node_type, NumSuperBlocks, NumMemoryBlocks, TileSize, SlabSize>;

  using allocator_type = slab_allocator_type;
  // using allocator_type = bump_allocator_type;

  GpuBTree::gpu_versioned_btree<key_type, value_type, branching_factor, allocator_type> tree;
  gpu_timer operations_timer;
  operations_timer.start_timer();
  tree.concurrent_insert_range(d_keys.data().get(),
                               d_values.data().get(),
                               num_keys,
                               d_find_keys_lower.data().get(),
                               d_find_keys_upper.data().get(),
                               num_ranges,
                               d_range_results.data().get(),
                               average_range_length);

  operations_timer.stop_timer();
  cuda_try(cudaDeviceSynchronize());
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto num_snapshots = tree.take_snapshot(stream);
  std::cout << "Snapshots count = " << num_snapshots + 1 << '\n';
  if (plot_tree) {
    std::cout << "Plotting..\n";
    tree.plot_dot("b-link-tree-links", true);
    tree.plot_dot("b-link-tree-nolinks", false);
    for (unsigned int i = 0; i <= num_snapshots; i++) {
      std::string dot_fname = "b-link-tree-nolinks_ts";
      dot_fname += std::to_string(i);
      tree.plot_dot(dot_fname, false, i);

      // links
      dot_fname = "b-link-tree-links_ts";
      dot_fname += std::to_string(i);
      tree.plot_dot(dot_fname, true, i);
    }
    // tree.print_vtree_nodes();
  }

  float operations_seconds = operations_timer.get_elapsed_s();
  auto memory_usage_gbs    = tree.compute_memory_usage();
  auto input_size_gbs      = double(sizeof(key_type) + sizeof(value_type)) * double(num_keys);
  input_size_gbs           = input_size_gbs / (1ull << 30);

  auto num_operations = num_keys + num_ranges;  // insertion + rq
  std::cout << "num_inserted: " << num_keys << std::endl;
  std::cout << "num_range_query: " << num_ranges << ", ";
  std::cout << "num_operations: " << num_operations << ", ";
  std::cout << "exist_ratio = " << exist_ratio * 100.0f << "%" << std::endl;
  std::cout << "memory_usage_gbs = " << memory_usage_gbs << ", ";
  std::cout << "input_size_gbs = " << input_size_gbs << " (";
  std::cout << double(memory_usage_gbs / input_size_gbs) << "x)" << std::endl;

  float operations_rate = float(num_keys + num_ranges) * float(1e-6) / operations_seconds;
  std::cout << "operations_rate = " << operations_rate << " mop/s";
  std::cout << " (" << operations_seconds << " s)" << std::endl;

  // copy-back the results:
  auto h_range_results = thrust::host_vector<pair_type>(d_range_results);
  // auto h_find_keys_upper = thrust::host_vector<key_type>(d_find_keys_upper);
  std::size_t prev_range_index = 0;

  if (plot_range) {
    std::size_t range_index = 0;
    std::cout << "Range " << range_index << ": ";
    std::cout << "[" << h_find_keys_lower[range_index] << ", ";
    std::cout << to_upper_bound(h_find_keys_lower[range_index]) << "): ";

    for (std::size_t index = 0; index < h_range_results.size(); index++) {
      range_index = index / average_range_length;
      auto r      = h_range_results[index];
      if (range_index != prev_range_index) {
        std::cout << '\n';
        std::cout << "Range " << range_index << ": ";
        std::cout << "[" << h_find_keys_lower[range_index] << ", ";
        // std::cout << h_find_keys_upper[range_index] << "): ";
        std::cout << to_upper_bound(h_find_keys_lower[range_index]) << "): ";

        prev_range_index = range_index;
      } else {
        if (r.first != std::numeric_limits<uint32_t>::max()) {
          std::cout << "{" << r.first << ", " << r.second << "}, ";
        }
      }
    }
  }

  /*auto num_errors =
      utils::validate(h_keys, h_find_keys, d_results, to_value, exist_ratio);*/

  // auto tree_size_gbs = tree.compute_memory_usage();
  // double pairs_size_gbs = double(num_keys) * sizeof(pair_type) / (1ull << 30);
  // std::cout << "B-Tree used: " << tree_size_gbs << " GiBs, "
  //          << "to store: " << pairs_size_gbs << " GiBs of pairs ("
  //          << pairs_size_gbs / tree_size_gbs * 100.0 << "%)" << std::endl;

  // if (validate) {
  //  tree.validate_tree_structure(h_keys, to_value);
  //}
  (void)(validate);
}
int main(int argc, char** argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);

  uint32_t num_keys   = get_arg_value<uint32_t>(arguments, "num-keys").value_or(32);
  uint32_t num_ranges = get_arg_value<uint32_t>(arguments, "num-ranges").value_or(num_keys);
  assert(num_ranges <= num_keys);

  int device_id             = get_arg_value<int>(arguments, "device").value_or(0);
  float exist_ratio         = get_arg_value<float>(arguments, "exist-ratio").value_or(1.0f);
  bool plot_tree            = get_arg_value<bool>(arguments, "plot").value_or(false);
  bool plot_range           = get_arg_value<bool>(arguments, "plot-range").value_or(false);
  bool validate             = get_arg_value<bool>(arguments, "validate").value_or(false);
  uint32_t branching_factor = get_arg_value<uint32_t>(arguments, "b").value_or(16);
  uint32_t average_range_length =
      get_arg_value<uint32_t>(arguments, "average-range-length").value_or(64);

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

  std::cout << "inserting: " << num_keys << std::endl;
  std::cout << "range query for: " << num_ranges << ", ";
  std::cout << "with average_range_length:" << average_range_length << ", ";

  using key_type   = uint32_t;
  using value_type = uint32_t;
  using size_type  = uint32_t;

  /*if (branching_factor == 32) {
    build<key_type, value_type, size_type, 32>(
        num_keys, exist_ratio, average_range_length, plot_tree, validate);
  } else */
  if (branching_factor == 16) {
    build<key_type, value_type, size_type, 16>(
        num_keys, num_ranges, exist_ratio, average_range_length, plot_tree, validate, plot_range);
  } else {
    std::cout << "Branching factor not supported." << std::endl;
    std::terminate();
  }
}
