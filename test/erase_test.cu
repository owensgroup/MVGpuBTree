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
#include <cmd.hpp>
#include <cstdint>
#include <gpu_timer.hpp>
#include <random>
#include <rkg.hpp>
#include <string>
#include <unordered_set>
#include <vector>

int main(int argc, char** argv) {
  auto arguments    = std::vector<std::string>(argv, argv + argc);
  uint32_t num_keys = get_arg_value<uint32_t>(arguments, "num_keys").value_or(64);
  int device_id     = get_arg_value<int>(arguments, "device").value_or(0);
  float erase_ratio = get_arg_value<float>(arguments, "erase_ratio").value_or(1.0f);
  uint32_t num_erase =
      get_arg_value<uint32_t>(arguments, "num_erase").value_or(erase_ratio * num_keys);
  float exist_ratio = get_arg_value<float>(arguments, "exist_ratio").value_or(1.0f);
  assert(num_erase <= num_keys);

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
  std::cout << "erasing: " << num_erase << ", ";
  std::cout << "with exist_ratio = " << exist_ratio * 100.0f << "%" << std::endl;

  // float exist_ratio = 1.0f;
  static constexpr int branching_factor = 32;

  using key_type                            = uint32_t;
  using value_type                          = uint32_t;
  using pair_type                           = pair_type<key_type, value_type>;
  static constexpr key_type invalid_key     = std::numeric_limits<uint32_t>::max();
  static constexpr value_type invalid_value = std::numeric_limits<uint32_t>::max();

  auto to_value = [] __host__ __device__(key_type x) { return x * 10; };

  auto d_keys   = thrust::device_vector<key_type>(num_keys, invalid_key);
  auto d_values = thrust::device_vector<value_type>(num_keys, invalid_value);

  auto h_erase_keys = std::vector<value_type>(num_keys, invalid_key);
  auto d_erase_keys = thrust::device_vector<value_type>(num_keys, invalid_key);
  auto d_results    = thrust::device_vector<value_type>(num_keys, invalid_value);

  unsigned seed = 0;
  std::random_device rd;
  std::mt19937 rng(seed);

  auto h_keys = rkg::generate_keys<key_type>(num_keys, rng, rkg::distribution_type::unique_random);
  rkg::prep_experiment_find_with_exist_ratio<key_type, value_type>(
      exist_ratio, num_keys, h_keys, h_erase_keys);
  h_keys.resize(num_keys);  // from num_keys * 2  => num_keys

  // copy to device
  d_keys       = h_keys;
  d_erase_keys = h_erase_keys;

  // assign values
  thrust::transform(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), to_value);

  GpuBTree::gpu_blink_tree<key_type, value_type, branching_factor> tree;

  gpu_timer insertion_timer;
  insertion_timer.start_timer();
  tree.insert(d_keys.data().get(), d_values.data().get(), num_keys);
  insertion_timer.stop_timer();

  gpu_timer erase_timer;
  erase_timer.start_timer();
  tree.erase(d_keys.data().get(), num_erase);
  erase_timer.start_timer();

  cuda_try(cudaDeviceSynchronize());

  float insertion_seconds = insertion_timer.get_elapsed_s();
  float erase_seconds     = erase_timer.get_elapsed_s();

  std::cout << "num_inserted: " << num_keys << std::endl;
  std::cout << "num_erase: " << num_erase << ", ";
  std::cout << "exist_ratio = " << exist_ratio * 100.0f << "%" << std::endl;

  float insertion_rate = float(num_keys) * float(1e-6) / insertion_seconds;
  std::cout << "insertion_rate = " << insertion_rate << " mop/s";
  std::cout << " (" << insertion_seconds << " s)" << std::endl;

  float erase_rate = float(num_erase) * float(1e-6) / erase_seconds;
  std::cout << "insertion_rate = " << erase_rate << " mop/s";
  std::cout << " (" << erase_seconds << " s)" << std::endl;
}
