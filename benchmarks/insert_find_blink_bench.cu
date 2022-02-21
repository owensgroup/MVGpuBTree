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

struct bench_rates {
  float insertion_rate;
  float find_rate;
};
template <typename BTree,
          bool supportsVersioning,
          typename KeyT,
          typename ValueT,
          typename SetT,
          typename Function>
bench_rates bench_versioned_insertion_find(thrust::device_vector<KeyT>& d_keys,
                                           thrust::device_vector<ValueT>& d_values,
                                           thrust::device_vector<KeyT>& d_queries,
                                           thrust::device_vector<ValueT>& d_results,
                                           std::vector<KeyT>& h_keys,
                                           std::vector<KeyT>& h_queries,
                                           float exist_ratio,
                                           bool in_place,
                                           bool validate_result,
                                           bool validate_tree_structure,
                                           const SetT& ref_set,
                                           std::size_t num_experiments,
                                           Function& to_value) {
  (void)in_place;
  (void)ref_set;
  (void)exist_ratio;
  cudaStream_t insertion_stream{0};
  cudaStream_t find_stream{0};
  float average_insertion_seconds(0.0f);
  float average_find_seconds(0.0f);

  for (std::size_t exp = 0; exp < num_experiments; exp++) {
    BTree tree;
    auto memory_usage = utils::compute_device_memory_usage();
    std::cout << "Using: " << double(memory_usage.used_bytes) / double(1 << 30) << " GiBs"
              << std::endl;

    gpu_timer insert_timer(insertion_stream);
    insert_timer.start_timer();
    if constexpr (supportsVersioning) {
      tree.insert(
          d_keys.data().get(), d_values.data().get(), d_keys.size(), insertion_stream, in_place);
    } else {
      tree.insert(d_keys.data().get(), d_values.data().get(), d_keys.size(), insertion_stream);
    }
    insert_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    average_insertion_seconds += insert_timer.get_elapsed_s();

    if (validate_tree_structure) { tree.validate_tree_structure(h_keys, to_value); }
    gpu_timer find_timer(find_stream);
    find_timer.start_timer();
    tree.find(d_queries.data().get(), d_results.data().get(), d_queries.size(), find_stream, false);
    find_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    average_find_seconds += find_timer.get_elapsed_s();

    if (validate_result) { utils::validate(ref_set, h_queries, d_results, to_value); }
  }

  average_insertion_seconds /= float(num_experiments);
  average_find_seconds /= float(num_experiments);

  float insertion_rate = float(d_keys.size()) / 1e6 / average_insertion_seconds;
  float find_rate      = float(d_queries.size()) / 1e6 / average_find_seconds;
  return {insertion_rate, find_rate};
}

int main(int argc, char** argv) {
  auto arguments    = std::vector<std::string>(argv, argv + argc);
  uint32_t num_keys = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1'000'000);
  int device_id     = get_arg_value<int>(arguments, "device").value_or(0);
  std::size_t num_experiments =
      get_arg_value<std::size_t>(arguments, "num-experiments").value_or(1llu);
  float exist_ratio      = get_arg_value<float>(arguments, "exist-ratio").value_or(1.0f);
  bool validate_result   = get_arg_value<bool>(arguments, "validate-result").value_or(false);
  bool validate_tree     = get_arg_value<bool>(arguments, "validate-tree").value_or(false);
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

  std::cout << "Generating input...\n";
  using key_type                            = uint32_t;
  using value_type                          = uint32_t;
  using pair_type                           = pair_type<key_type, value_type>;
  static constexpr key_type invalid_key     = std::numeric_limits<key_type>::max();
  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();
  auto to_value                             = [] __host__ __device__(key_type x) { return x % 10; };

  unsigned seed = 0;
  std::random_device rd;
  std::mt19937 rng(seed);

  // device vectors
  auto d_keys      = thrust::device_vector<key_type>(num_keys, invalid_key);
  auto d_values    = thrust::device_vector<value_type>(num_keys, invalid_value);
  auto d_find_keys = thrust::device_vector<key_type>(num_keys, invalid_key);
  auto d_results   = thrust::device_vector<value_type>(num_keys, invalid_value);

  // host vectors
  auto h_find_keys = std::vector<value_type>(num_keys, invalid_key);
  auto h_keys =
      rkg::generate_keys<key_type>(num_keys * 2, rng, rkg::distribution_type::unique_random);

  rkg::prep_experiment_find_with_exist_ratio<key_type, value_type>(
      exist_ratio, num_keys, h_keys, h_find_keys);
  h_keys.resize(num_keys);  // from num_keys * 2  => num_keys

  // copy to device
  d_keys      = h_keys;
  d_find_keys = h_find_keys;

  // assign values
  thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), to_value);

  std::unordered_set<key_type> cpu_ref_set;
  if (validate_result) { cpu_ref_set.insert(h_keys.begin(), h_keys.end()); }

  std::cout << "Benchmarking...\n";
  std::cout << "num_keys = " << num_keys << ',';
  std::cout << "exist_ratio = " << exist_ratio << '\n';
  static constexpr int branching_factor = 16;
  using node_type             = GpuBTree::node_type<key_type, value_type, branching_factor>;
  using slab_allocator_type   = device_allocator::SlabAllocLight<node_type, 8, 1024 * 8, 16, 128>;
  using bump_allocator_type   = device_bump_allocator<node_type>;
  using bump_allocator_type_4 = device_bump_allocator<node_type, 33554432>;

  using blink_tree_slab_type =
      GpuBTree::gpu_blink_tree<key_type, value_type, branching_factor, slab_allocator_type>;
  using blink_tree_bump_type =
      GpuBTree::gpu_blink_tree<key_type, value_type, branching_factor, bump_allocator_type>;
  using blink_tree_bump4_type =
      GpuBTree::gpu_blink_tree<key_type, value_type, branching_factor, bump_allocator_type_4>;

  using vblink_tree_slab_type =
      GpuBTree::gpu_versioned_btree<key_type, value_type, branching_factor, slab_allocator_type>;
  using vblink_tree_bump_type =
      GpuBTree::gpu_versioned_btree<key_type, value_type, branching_factor, bump_allocator_type>;

  std::string report_dir = output_dir + '/' + device_name + "/blink/";
  std::filesystem::create_directories(report_dir);

  std::string filename   = "rates.csv";
  bool output_file_exist = std::filesystem::exists(report_dir + filename);
  std::fstream result_output(report_dir + filename, std::ios::app);
  if (!output_file_exist) {
    result_output << "num_keys" << ',';
    result_output << "num_queries" << ',';
    result_output << "num_experiments" << ',';
    result_output << "exist_ratio" << ',';

    result_output << "blink_slab_insert" << ',';
    result_output << "blink_slab_find" << ',';
    result_output << "blink_bump_insert" << ',';
    result_output << "blink_bump_find" << ',';

    result_output << "blink_bump4_insert" << ',';
    result_output << "blink_bump4_find" << ',';
    result_output << '\n';
  }

  result_output << num_keys << ',';
  result_output << num_keys << ',';
  result_output << num_experiments << ',';
  result_output << exist_ratio << ',';

  {
    auto rates = bench_versioned_insertion_find<blink_tree_slab_type, false>(d_keys,
                                                                             d_values,
                                                                             d_find_keys,
                                                                             d_results,
                                                                             h_keys,
                                                                             h_find_keys,
                                                                             exist_ratio,
                                                                             false,
                                                                             validate_result,
                                                                             validate_tree,
                                                                             cpu_ref_set,
                                                                             num_experiments,
                                                                             to_value);
    result_output << rates.insertion_rate << ',';
    result_output << rates.find_rate << ',';
  }
  {
    auto rates = bench_versioned_insertion_find<blink_tree_bump_type, false>(d_keys,
                                                                             d_values,
                                                                             d_find_keys,
                                                                             d_results,
                                                                             h_keys,
                                                                             h_find_keys,
                                                                             exist_ratio,
                                                                             false,
                                                                             validate_result,
                                                                             validate_tree,
                                                                             cpu_ref_set,
                                                                             num_experiments,
                                                                             to_value);
    result_output << rates.insertion_rate << ',';
    result_output << rates.find_rate << ',';
  }
  {
    auto rates = bench_versioned_insertion_find<blink_tree_bump4_type, false>(d_keys,
                                                                              d_values,
                                                                              d_find_keys,
                                                                              d_results,
                                                                              h_keys,
                                                                              h_find_keys,
                                                                              exist_ratio,
                                                                              false,
                                                                              validate_result,
                                                                              validate_tree,
                                                                              cpu_ref_set,
                                                                              num_experiments,
                                                                              to_value);
    result_output << rates.insertion_rate << ',';
    result_output << rates.find_rate << ',';
  }
  result_output << '\n';
}
