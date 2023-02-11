/*
 *   Copyright 2023 The Regents of the University of California, Davis
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

#include <device_bump_allocator.hpp>

#include <thrust/sequence.h>
#include <cmd.hpp>
#include <vector>

int main(int argc, char** argv) {
  auto arguments    = std::vector<std::string>(argv, argv + argc);
  uint32_t num_keys = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1'000'000);
  bool validate     = get_arg_value<bool>(arguments, "validate").value_or(true);

  static constexpr uint32_t branching_factor = 16;

  using key_type   = uint32_t;
  using value_type = uint32_t;
  using node_type  = GpuBTree::node_type<key_type, value_type, branching_factor>;

  static constexpr std::size_t two_gibs = (std::size_t{1} << 30) / sizeof(node_type);

  using bump_allocator_type = device_bump_allocator<node_type, two_gibs>;

  auto invalid_key   = GpuBTree::gpu_blink_tree<key_type, value_type>::invalid_key;
  auto invalid_value = GpuBTree::gpu_blink_tree<key_type, value_type>::invalid_value;

  auto d_keys   = thrust::device_vector<key_type>(num_keys, invalid_key);
  auto d_values = thrust::device_vector<value_type>(num_keys, invalid_value);

  thrust::sequence(d_keys.begin(), d_keys.end(), key_type{0});

  auto to_value = [] __host__ __device__(key_type x) { return x * 10; };
  thrust::transform(d_keys.begin(), d_keys.end(), d_values.begin(), to_value);

  using tree_type =
      GpuBTree::gpu_blink_tree<key_type, value_type, branching_factor, bump_allocator_type>;

  const bool input_is_sorted = true;
  tree_type tree(d_keys.data().get(), d_values.data().get(), num_keys, input_is_sorted);
}
