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

#include <cuda_profiler_api.h>
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

struct bench_rates {
  float insertion_rate;
  float ops_rate;
};
template <typename BTree,
          bool supportsVersioning,
          typename KeyT,
          typename ValueT,
          typename PairT,
          typename SetT,
          typename SizeT,
          typename Function0,
          typename Function1>
bench_rates bench_versioned_insert_range(thrust::device_vector<KeyT> &d_keys,
                                         thrust::device_vector<ValueT> &d_values,
                                         SizeT initial_tree_size,
                                         thrust::device_vector<KeyT> &d_lower_bound,
                                         thrust::device_vector<KeyT> &d_upper_bound,
                                         thrust::device_vector<PairT> &d_results,
                                         SizeT average_range_length,
                                         std::vector<KeyT> &h_keys,
                                         std::vector<KeyT> &h_lower_bound,
                                         bool in_place,
                                         bool validate_result,
                                         bool validate_tree_structure,
                                         const SetT &ref_set_v0,
                                         const SetT &ref_set_v1,
                                         std::size_t num_experiments,
                                         Function0 &to_value,
                                         Function1 &to_upper_bound) {
  (void)in_place;
  (void)ref_set_v0;
  (void)ref_set_v1;
  cudaStream_t insertion_stream{0};
  cudaStream_t concurrent_ops_stream{0};
  float average_insertion_seconds(0.0f);
  float average_ops_seconds(0.0f);

  const KeyT invalid_key     = std::numeric_limits<KeyT>::max();
  const ValueT invalid_value = std::numeric_limits<ValueT>::max();
  const PairT invalid_pair(invalid_key, invalid_value);

  auto second_insertion_batch_size = d_keys.size() - initial_tree_size;

  for (std::size_t exp = 0; exp < num_experiments; exp++) {
    std::cout << "Experiment " << exp << "/" << num_experiments << "...";
    std::cout << std::endl;

    BTree tree;
    gpu_timer insert_timer(insertion_stream);
    insert_timer.start_timer();
    if constexpr (supportsVersioning) {
      tree.insert(d_keys.data().get(),
                  d_values.data().get(),
                  initial_tree_size,
                  insertion_stream,
                  in_place);
    } else {
      tree.insert(d_keys.data().get(), d_values.data().get(), initial_tree_size, insertion_stream);
    }
    insert_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    average_insertion_seconds += insert_timer.get_elapsed_s();

    if (validate_tree_structure) {
      std::vector<KeyT> h_keys_v0(h_keys.begin(), h_keys.begin() + initial_tree_size);
      tree.validate_tree_structure(h_keys_v0, to_value);
      std::cout << "Validation success @v0" << std::endl;
    }

    cuda_try(cudaProfilerStart());

    gpu_timer ops_timer(concurrent_ops_stream);
    ops_timer.start_timer();
    tree.concurrent_insert_range(d_keys.data().get() + initial_tree_size,
                                 d_values.data().get() + initial_tree_size,
                                 second_insertion_batch_size,
                                 d_lower_bound.data().get(),
                                 d_upper_bound.data().get(),
                                 d_lower_bound.size(),
                                 d_results.data().get(),
                                 average_range_length,
                                 concurrent_ops_stream);
    ops_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    average_ops_seconds += ops_timer.get_elapsed_s();
    cuda_try(cudaProfilerStop());

    if (validate_result) {
      std::cout << "Validating... ";
      utils::validate_concurrent_ops(h_lower_bound,
                                     d_results,
                                     average_range_length,
                                     ref_set_v0,
                                     ref_set_v1,
                                     to_value,
                                     to_upper_bound);
      thrust::fill(thrust::device, d_results.begin(), d_results.end(), invalid_pair);
    }
    if (validate_tree_structure) {
      std::vector<KeyT> h_keys_v1(h_keys.begin(), h_keys.end());
      tree.validate_tree_structure(h_keys_v1, to_value);
      std::cout << "Validation success @v1" << std::endl;
    }
  }

  average_insertion_seconds /= float(num_experiments);
  average_ops_seconds /= float(num_experiments);

  float insertion_rate = float(initial_tree_size) / 1e6 / average_insertion_seconds;
  float ops_rate =
      float(second_insertion_batch_size + d_lower_bound.size()) / 1e6 / average_ops_seconds;

  std::cout << "build_rate: " << insertion_rate << std::endl;
  std::cout << "concurrent_ops_rate: " << ops_rate << std::endl;

  return {insertion_rate, ops_rate};
}

int main(int argc, char **argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);

  uint32_t initial_tree_size =
      get_arg_value<uint32_t>(arguments, "initial-size").value_or(1'000'000);
  uint32_t num_operations       = get_arg_value<uint32_t>(arguments, "num-ops").value_or(1'000'000);
  float update_ratio            = get_arg_value<float>(arguments, "update-ratio").value_or(0.5f);
  uint32_t average_range_length = get_arg_value<uint32_t>(arguments, "range-length").value_or(16);

  int device_id = get_arg_value<int>(arguments, "device").value_or(0);
  std::size_t num_experiments =
      get_arg_value<std::size_t>(arguments, "num-experiments").value_or(1llu);
  bool validate        = get_arg_value<bool>(arguments, "validate").value_or(false);
  bool validate_result = get_arg_value<bool>(arguments, "validate-result").value_or(validate);
  bool validate_tree   = get_arg_value<bool>(arguments, "validate-tree").value_or(validate);

  std::string output_dir = get_arg_value<std::string>(arguments, "output-dir").value_or("./");

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

  std::string device_name(devProp.name);
  std::replace(device_name.begin(), device_name.end(), ' ', '-');

  uint32_t num_insertions  = num_operations * update_ratio;
  uint32_t num_range_query = num_operations - num_insertions;
  uint32_t num_keys        = num_insertions + initial_tree_size;

  std::cout << "Benchmarking...\n";
  std::cout << std::boolalpha;
  std::cout << "initial_tree_size = " << initial_tree_size << ",\n";
  std::cout << "num_operations = " << num_operations << ", ";
  std::cout << "num_insertions = " << num_insertions << ", ";
  std::cout << "num_range_query = " << num_range_query << ", ";
  std::cout << "average_range_length = " << average_range_length << "\n";
  std::cout << "update_ratio = " << update_ratio << ", ";
  std::cout << "num_experiments = " << num_experiments << ", \n";
  std::cout << "validate_tree = " << validate_tree << ", ";
  std::cout << "validate_result = " << validate_result << "\n";

  std::cout << "------------------------\n";
  std::cout << "Generating input...\n";

  using key_type                 = uint32_t;
  using value_type               = uint32_t;
  using pair_type                = pair_type<key_type, value_type>;
  const key_type invalid_key     = std::numeric_limits<key_type>::max();
  const value_type invalid_value = std::numeric_limits<value_type>::max();
  const pair_type invalid_pair(invalid_key, invalid_value);
  auto to_value       = [] __host__ __device__(key_type x) { return x % 10; };
  auto to_upper_bound = [average_range_length] __host__ __device__(key_type x) {
    return x + average_range_length;
  };

  unsigned seed = 0;
  std::random_device rd;
  std::mt19937_64 rng(seed);
  // device vectors
  auto d_keys   = thrust::device_vector<key_type>(num_keys, invalid_key);
  auto d_values = thrust::device_vector<value_type>(num_keys, invalid_value);

  auto d_range_lower = thrust::device_vector<key_type>(num_range_query, invalid_key);
  auto d_range_upper = thrust::device_vector<key_type>(num_range_query, invalid_key);
  auto d_range_results =
      thrust::device_vector<pair_type>(num_range_query * average_range_length, invalid_pair);

  // host vectors
  auto h_keys = rkg::generate_keys<key_type>(num_keys, rng, rkg::distribution_type::unique_random);
  auto h_range_lower = std::vector<key_type>(num_range_query, invalid_key);

  rkg::prep_experiment_range_query(h_keys, initial_tree_size, h_range_lower, num_range_query, rng);

  // copy to device
  d_keys        = h_keys;
  d_range_lower = h_range_lower;

  // assign values and upper bound
  thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), to_value);
  thrust::transform(thrust::device,
                    d_range_lower.begin(),
                    d_range_lower.end(),
                    d_range_upper.begin(),
                    to_upper_bound);

  std::set<key_type> cpu_ref_set_v0;  // contains initial tree
  std::set<key_type> cpu_ref_set_v1;  // contains inserted keys
  if (validate_result) {
    std::cout << "Building CPU reference sets...\n";
    cpu_ref_set_v0.insert(h_keys.begin(), h_keys.begin() + initial_tree_size);
    cpu_ref_set_v1.insert(h_keys.begin() + initial_tree_size, h_keys.end());
  }

  static constexpr int branching_factor = 16;
  using node_type           = GpuBTree::node_type<key_type, value_type, branching_factor>;
  using slab_allocator_type = device_allocator::SlabAllocLight<node_type, 8, 128 * 64, 16, 128>;
  using bump_allocator_type = device_bump_allocator<node_type>;

  using blink_tree_slab_type =
      GpuBTree::gpu_blink_tree<key_type, value_type, branching_factor, slab_allocator_type>;
  using blink_tree_bump_type =
      GpuBTree::gpu_blink_tree<key_type, value_type, branching_factor, bump_allocator_type>;

  using vblink_tree_slab_type =
      GpuBTree::gpu_versioned_btree<key_type, value_type, branching_factor, slab_allocator_type>;
  using vblink_tree_bump_type =
      GpuBTree::gpu_versioned_btree<key_type, value_type, branching_factor, bump_allocator_type>;

  std::string report_dir = output_dir + '/' + device_name + "/versioned_insert_range/";
  std::filesystem::create_directories(report_dir);

  std::string filename = "rates_initial" + std::to_string(int(initial_tree_size / 1e6)) +
                         "M_update" + std::to_string(int(update_ratio * 100)) + "_range_length" +
                         std::to_string(average_range_length) + ".csv";
  bool output_file_exist = std::filesystem::exists(report_dir + filename);
  std::fstream result_output(report_dir + filename, std::ios::app);
  if (!output_file_exist) {
    result_output << "initial_tree_size" << ',';
    result_output << "num_insertions" << ',';
    result_output << "num_range_query" << ',';
    result_output << "num_experiments" << ',';

    result_output << "vblink_slab_out_of_place_insert" << ',';
    result_output << "vblink_slab_out_of_place_concurrent_ops" << ',';

    result_output << "blink_slab_insert" << ',';
    result_output << "blink_slab_concurrent_ops" << ',';
    result_output << '\n';
  }

  result_output << initial_tree_size << ',';
  result_output << num_insertions << ',';
  result_output << num_range_query << ',';
  result_output << num_experiments << ',';

  std::cout << "Running experiment...\n";
  {
    auto rates = bench_versioned_insert_range<vblink_tree_slab_type, true>(d_keys,
                                                                           d_values,
                                                                           initial_tree_size,
                                                                           d_range_lower,
                                                                           d_range_upper,
                                                                           d_range_results,
                                                                           average_range_length,
                                                                           h_keys,
                                                                           h_range_lower,
                                                                           true,
                                                                           validate_result,
                                                                           validate_tree,
                                                                           cpu_ref_set_v0,
                                                                           cpu_ref_set_v1,
                                                                           num_experiments,
                                                                           to_value,
                                                                           to_upper_bound);

    result_output << rates.insertion_rate << ',';
    result_output << rates.ops_rate << ',';
  }

  {
    auto rates = bench_versioned_insert_range<blink_tree_slab_type, false>(d_keys,
                                                                           d_values,
                                                                           initial_tree_size,
                                                                           d_range_lower,
                                                                           d_range_upper,
                                                                           d_range_results,
                                                                           average_range_length,
                                                                           h_keys,
                                                                           h_range_lower,
                                                                           true,
                                                                           validate_result,
                                                                           validate_tree,
                                                                           cpu_ref_set_v0,
                                                                           cpu_ref_set_v1,
                                                                           num_experiments,
                                                                           to_value,
                                                                           to_upper_bound);

    result_output << rates.insertion_rate << ',';
    result_output << rates.ops_rate << ',';
  }

  result_output << '\n';
}
