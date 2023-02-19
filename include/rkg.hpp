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
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <pair_type.hpp>
#include <set>
#include <string>
#include <string_view>
#include <typeinfo>
#include <unordered_set>

namespace rkg {

template <typename key_type, typename value_type, typename size_type>
inline void prep_experiment_find_with_exist_ratio(float exist_ratio,
                                                  size_type num_keys,
                                                  const std::vector<key_type>& keys,
                                                  std::vector<key_type>& queries) {
  if (exist_ratio < 1.0f) { assert(num_keys * 2 == keys.size()); }
  unsigned int end_index   = static_cast<unsigned int>(num_keys * (-exist_ratio + 2));
  unsigned int start_index = end_index - num_keys;

  // Need to copy our range [start_index, end_index) from keys into queries.
  std::copy(keys.begin() + start_index, keys.begin() + end_index, queries.begin());
}

template <typename key_type, typename rng_type>
inline void prep_experiment_range_query(const std::vector<key_type>& keys,
                                        const std::size_t num_keys,
                                        std::vector<key_type>& queries,
                                        const std::size_t num_rqs,
                                        rng_type& rng) {
  // copy all keys
  queries.resize(std::max(num_keys, num_rqs));
  std::copy(keys.begin(), keys.begin() + num_keys, queries.begin());

  // sample the rest
  if (num_rqs > num_keys) {
    auto current_rqs = num_keys;
    while (current_rqs < num_rqs) {
      auto sampled         = rng() % num_keys;
      queries[current_rqs] = keys[sampled];
      current_rqs++;
    }
  }
  // shuffle
  std::shuffle(queries.begin(), queries.end(), rng);

  // bring size back to required number of queries
  queries.resize(num_rqs);
}

enum class distribution_type { unique_ascending, unique_descending, unique_random, has_duplicates };

template <typename key_type, typename rng_type>
inline std::vector<key_type> generate_keys(const uint32_t num_keys,
                                           rng_type& rng,
                                           const distribution_type dist,
                                           const float duplicates_ratio = 1.0f) {
  std::vector<key_type> keys(num_keys, 1);
  std::iota(keys.begin(), keys.end(), 1);
  uint32_t num_unique = 0;
  switch (dist) {
    case distribution_type::unique_random: std::shuffle(keys.begin(), keys.end(), rng); break;
    case distribution_type::unique_ascending: break;
    case distribution_type::unique_descending: std::reverse(keys.begin(), keys.end()); break;
    case distribution_type::has_duplicates:
      num_unique = static_cast<uint32_t>(static_cast<double>(num_keys) * duplicates_ratio);
    default: break;
  }
  if (dist == distribution_type::has_duplicates) {
    for (size_t key_id = 0; key_id < num_keys; key_id++) { keys[key_id] = rng() % num_unique + 1; }
  }
  return keys;
}

template <typename key_type>
inline thrust::device_vector<key_type> generate_random_keys(const uint32_t num_keys,
                                                    const int seed,
                                                    const bool cache = false){
  std::string dataset_dir = "dataset";
  std::string dataset_name = std::to_string(num_keys) + "_" + std::to_string(seed);
  std::string dataset_path = dataset_dir + "/" + dataset_name;

  std::vector<key_type> keys(num_keys);

  if (cache) {
    if (std::filesystem::exists(dataset_dir)) {
      if (std::filesystem::exists(dataset_path)) {
        std::cout << "Reading cached keys.." << std::endl;
        std::ifstream dataset(dataset_path, std::ios::binary);
        dataset.read((char*)keys.data(), sizeof(key_type) * num_keys);
        dataset.close();
        thrust::device_vector<key_type> d_keys(keys);
        return std::move(d_keys);
      }
    } else {
      std::filesystem::create_directory(dataset_dir);
    }
  }


  key_type min_usable_key = 1;
  key_type max_usable_key = std::numeric_limits<key_type>::max() - 2;

  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<key_type> key_dist(min_usable_key, max_usable_key);

  std::unordered_set<key_type> build_keys_set;
  while (build_keys_set.size() < num_keys) {
    key_type key = key_dist(gen);
    build_keys_set.insert(key);
  }

  std::copy(build_keys_set.begin(), build_keys_set.end(), keys.begin());
  thrust::device_vector<key_type> d_keys(keys);
  thrust::sort(d_keys.begin(), d_keys.end());

  if (cache) {
    std::cout << "Caching.." << std::endl;
    std::ofstream dataset(dataset_path, std::ios::binary);
    thrust::host_vector<key_type> sorted_keys(d_keys);
    dataset.write((char*)sorted_keys.data(), sizeof(key_type) * num_keys);
    dataset.close();
  }

  return std::move(d_keys);
}
}  // namespace rkg
