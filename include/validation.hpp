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
#include <functional>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <typeinfo>
#include <unordered_set>

#include <pair_type.hpp>

namespace utils {

template <typename key_type, typename value_type, typename function>
inline uint64_t validate(const std::vector<key_type>& h_keys,
                         const std::vector<key_type>& h_find_keys,
                         const thrust::device_vector<value_type>& d_results,
                         function to_value,
                         float exist_ratio = 1.0f) {
  uint64_t num_errors = 0;
  uint64_t max_errors = 10;
  using pair_type     = pair_type<key_type, value_type>;
  auto h_results      = thrust::host_vector<value_type>(d_results);
  std::unordered_set<key_type> cpu_ref_set;
  if (exist_ratio != 1.0f) { cpu_ref_set.insert(h_keys.begin(), h_keys.end()); }
  for (size_t i = 0; i < h_results.size(); i++) {
    key_type query_key         = h_find_keys[i];
    value_type query_result    = h_results[i];
    value_type expected_result = to_value(query_key);
    if (exist_ratio != 1.0f) {
      auto expected_result_ptr = cpu_ref_set.find(query_key);
      if (expected_result_ptr == cpu_ref_set.end()) {
        expected_result = std::numeric_limits<uint32_t>::max();
      }
    }

    if (query_result != expected_result) {
      std::string message = std::string("query_key = ") + std::to_string(query_key) +
                            std::string(", expected: ") + std::to_string(expected_result) +
                            std::string(", found: ") + std::to_string(query_result);
      std::cout << message << std::endl;
      num_errors++;
      if (num_errors == max_errors) break;
    }
  }
  return num_errors;
}

template <typename key_type, typename value_type, typename Function>
inline uint64_t validate(const std::unordered_set<key_type>& cpu_ref_set,
                         const std::vector<key_type>& h_find_keys,
                         const thrust::device_vector<value_type>& d_results,
                         Function to_value) {
  uint64_t num_errors = 0;
  uint64_t max_errors = 10;
  using pair_type     = pair_type<key_type, value_type>;
  auto h_results      = thrust::host_vector<value_type>(d_results);
  for (size_t i = 0; i < h_results.size(); i++) {
    key_type query_key         = h_find_keys[i];
    value_type query_result    = h_results[i];
    value_type expected_result = to_value(query_key);
    auto expected_result_ptr   = cpu_ref_set.find(query_key);
    if (expected_result_ptr == cpu_ref_set.end()) {
      expected_result = std::numeric_limits<uint32_t>::max();
    }

    if (query_result != expected_result) {
      std::string message = std::string("query_key = ") + std::to_string(query_key) +
                            std::string(", expected: ") + std::to_string(expected_result) +
                            std::string(", found: ") + std::to_string(query_result);
      std::cout << message << std::endl;
      num_errors++;
      if (num_errors == max_errors) break;
    }
  }
  if (num_errors == 0) { std::cout << "Ok" << std::endl; }
  return num_errors;
}

template <typename key_type, typename value_type, typename Function>
inline uint64_t validate_concurrent_find_erase(const std::unordered_set<key_type>& erased_keys,
                                               const std::vector<key_type>& h_find_keys,
                                               const thrust::device_vector<value_type>& d_results,
                                               Function to_value) {
  uint64_t num_errors          = 0;
  uint64_t max_errors          = 10;
  using pair_type              = pair_type<key_type, value_type>;
  auto h_results               = thrust::host_vector<value_type>(d_results);
  value_type empty_value       = std::numeric_limits<value_type>::max();
  std::size_t read_erased_keys = 0;
  for (size_t i = 0; i < h_results.size(); i++) {
    key_type query_key         = h_find_keys[i];
    value_type query_result    = h_results[i];
    value_type expected_result = to_value(query_key);

    if (query_result != expected_result) {
      // check if was erased
      auto key_was_erased_ptr = erased_keys.find(query_key);
      bool key_was_erased     = key_was_erased_ptr != erased_keys.end();
      bool key_was_found      = query_result != empty_value;
      read_erased_keys++;
      if ((!key_was_erased) || (key_was_erased && key_was_found)) {
        std::string message = std::string("query_key = ") + std::to_string(query_key) +
                              std::string(", expected: ") + std::to_string(expected_result) +
                              std::string(" (or erased), found: ") + std::to_string(query_result);
        std::cout << message << std::endl;
        num_errors++;
        if (num_errors == max_errors) break;
      }
    }
  }
  if (num_errors == 0) {
    std::cout << "Read " << read_erased_keys << " out of " << h_results.size()
              << " queries from v1";
    std::cout << " (" << double(read_erased_keys) / double(h_results.size()) * 100.0
              << " % from queries - ";
    std::cout << double(read_erased_keys) / double(erased_keys.size()) * 100.0
              << " % from erased keys)\n";

    std::cout << "Ok" << std::endl;
  }
  return num_errors;
}

template <typename key_type, typename value_type, typename function>
inline uint64_t validate(const std::vector<key_type>& h_keys,
                         const uint32_t& num_keys,
                         const std::vector<key_type>& h_find_keys,
                         const thrust::device_vector<value_type>& d_results,
                         function to_value) {
  uint64_t num_errors = 0;
  uint64_t max_errors = 10;
  using pair_type     = pair_type<key_type, value_type>;
  auto h_results      = thrust::host_vector<value_type>(d_results);
  std::unordered_set<key_type> cpu_ref_set;
  cpu_ref_set.insert(h_keys.begin(), h_keys.begin() + num_keys);
  for (size_t i = 0; i < h_results.size(); i++) {
    key_type query_key         = h_find_keys[i];
    value_type query_result    = h_results[i];
    value_type expected_result = to_value(query_key);
    auto expected_result_ptr   = cpu_ref_set.find(query_key);
    if (expected_result_ptr == cpu_ref_set.end()) {
      expected_result = std::numeric_limits<uint32_t>::max();
    }

    if (query_result != expected_result) {
      std::string message = std::string("query_key = ") + std::to_string(query_key) +
                            std::string(", expected: ") + std::to_string(expected_result) +
                            std::string(", found: ") + std::to_string(query_result);
      std::cout << message << std::endl;
      num_errors++;
      if (num_errors == max_errors) break;
    }
  }
  return num_errors;
}

template <typename key_type,
          typename pair_type,
          typename size_type,
          typename function0,
          typename function1>
inline uint64_t validate(const std::vector<key_type>& h_keys,
                         const std::vector<key_type>& h_range_keys_lower,
                         const thrust::device_vector<pair_type>& d_range_results,
                         const thrust::device_vector<size_type>& d_count_results,
                         const size_type& average_range_length,
                         function0 to_value,
                         function1 to_upper_bound) {
  static constexpr key_type invalid_key     = std::numeric_limits<uint32_t>::max();
  using value_type                          = typename pair_type::value_type;
  static constexpr value_type invalid_value = std::numeric_limits<uint32_t>::max();

  uint64_t num_errors  = 0;
  uint64_t max_errors  = 10;
  auto h_range_results = thrust::host_vector<pair_type>(d_range_results);
  auto h_count_results = thrust::host_vector<size_type>(d_count_results);

  std::set<key_type> ref_set(h_keys.begin(), h_keys.end());
  bool has_range = h_range_results.size() != 0;
  bool has_count = h_count_results.size() != 0;

  for (size_t i = 0; i < h_range_keys_lower.size(); i++) {
    key_type query_lower_bound = h_range_keys_lower[i];
    key_type query_upper_bound = to_upper_bound(query_lower_bound);

    auto lower_bound             = ref_set.lower_bound(query_lower_bound);
    size_type expected_count     = 0;
    size_type range_start_offset = i * average_range_length;
    size_type result_offset      = range_start_offset;
    while (lower_bound != ref_set.end() && (*lower_bound) >= query_lower_bound &&
           (*lower_bound) < query_upper_bound) {
      auto expected_key   = *lower_bound;
      auto expected_value = to_value(expected_key);
      auto expected_pair  = pair_type(expected_key, expected_value);

      if (has_range && (expected_pair != h_range_results[result_offset])) {
        std::string message =
            std::string("query_range: [") + std::to_string(query_lower_bound) + std::string(", ") +
            std::to_string(query_upper_bound) + std::string(") ") + std::string(", expected: {") +
            std::to_string(expected_key) + std::string(", ") + std::to_string(expected_key) +
            std::string("} ") + std::string(", found: {") +
            std::to_string(h_range_results[result_offset].first) + std::string(", ") +
            std::to_string(h_range_results[result_offset].second) + std::string("} ");
        std::cout << message << std::endl;
        num_errors++;
        if (num_errors == max_errors) break;
      }
      expected_count++;
      lower_bound++;
      result_offset++;
    }

    if (has_range && result_offset != (range_start_offset + average_range_length) &&
        result_offset < h_range_results.size()) {
      auto last_pair = h_range_results[result_offset];
      if (last_pair != pair_type(invalid_key, invalid_value)) {
        std::string message =
            std::string("query_range: [") + std::to_string(query_lower_bound) + std::string(", ") +
            std::to_string(query_upper_bound) + std::string(") ") +
            std::string(", expected an empty pair, found: {") + std::to_string(last_pair.first) +
            std::string(", ") + std::to_string(last_pair.second) + std::string("} ");
        std::cout << message << std::endl;
      }
    }
    if (has_count && (expected_count != h_count_results[i])) {
      std::string message = std::string("query_range: [") + std::to_string(query_lower_bound) +
                            std::string(", ") + std::to_string(query_upper_bound) +
                            std::string(") ") + std::string(", expected count:") +
                            std::to_string(expected_count) + std::string(" , found count:") +
                            std::to_string(h_count_results[i]);
      std::cout << message << std::endl;
      num_errors++;
    }
    if (num_errors == max_errors) break;
  }
  return num_errors;
}

template <typename key_type,
          typename pair_type,
          typename size_type,
          typename function0,
          typename function1>
uint64_t validate_concurrent_ops(const std::vector<key_type>& h_range_keys_lower,
                                 const thrust::device_vector<pair_type>& d_range_results,
                                 const size_type& average_range_length,
                                 const std::set<key_type>& ref_set_v0,
                                 const std::set<key_type>& ref_set_v1,
                                 function0 to_value,
                                 function1 to_upper_bound) {
  static constexpr key_type invalid_key     = std::numeric_limits<uint32_t>::max();
  using value_type                          = typename pair_type::value_type;
  static constexpr value_type invalid_value = std::numeric_limits<uint32_t>::max();

  uint64_t num_errors  = 0;
  uint64_t max_errors  = 1;
  auto h_range_results = thrust::host_vector<pair_type>(d_range_results);

  std::vector<key_type> v1_found_keys;

  for (size_t i = 0; i < h_range_keys_lower.size(); i++) {
    key_type query_lower_bound = h_range_keys_lower[i];
    key_type query_upper_bound = to_upper_bound(query_lower_bound);

    auto lower_bound                   = ref_set_v0.lower_bound(query_lower_bound);
    const size_type range_start_offset = i * average_range_length;
    size_type result_offset            = range_start_offset;
    auto last_v0_key                   = *lower_bound;
    while (lower_bound != ref_set_v0.end() && (*lower_bound) >= query_lower_bound &&
           (*lower_bound) < query_upper_bound) {
      auto expected_key   = *lower_bound;
      auto expected_value = to_value(expected_key);
      auto expected_pair  = pair_type(expected_key, expected_value);
      auto found_pair     = h_range_results[result_offset];

      last_v0_key = *lower_bound;
      lower_bound++;

      if (expected_pair != found_pair) {
        lower_bound--;  // go back
        auto v1_location = ref_set_v1.find(found_pair.first);
        if (v1_location == ref_set_v1.end()) {
          std::string message =
              std::string("v0: query_range: [") + std::to_string(query_lower_bound) +
              std::string(", ") + std::to_string(query_upper_bound) + std::string(") ") +
              std::string(", expected: {") + std::to_string(expected_key) + std::string(", ") +
              std::to_string(expected_key) + std::string("} ") + std::string(", found: {") +
              std::to_string(found_pair.first) + std::string(", ") +
              std::to_string(found_pair.second) + std::string("} ");
          std::cout << message << std::endl;
          num_errors++;
          if (num_errors == max_errors) break;
        } else {
          v1_found_keys.push_back(expected_key);
        }
      }
      result_offset++;
    }
    // continue from the second map
    lower_bound        = ref_set_v1.lower_bound(last_v0_key);
    auto offset_before = result_offset;
    while (lower_bound != ref_set_v1.end() && (*lower_bound) >= query_lower_bound &&
           (*lower_bound) < query_upper_bound) {
      auto expected_key   = *lower_bound;
      auto expected_value = to_value(expected_key);
      auto expected_pair  = pair_type(expected_key, expected_value);
      auto found_pair     = h_range_results[result_offset];

      if (expected_pair != found_pair) {
        // means key was not inserted when the RQ ran
      } else {
        result_offset++;
      }
      lower_bound++;
    }

    if (result_offset != (range_start_offset + average_range_length) &&
        result_offset < h_range_results.size()) {
      auto last_pair = h_range_results[result_offset];
      if (last_pair != pair_type(invalid_key, invalid_value)) {
        std::cout << *(ref_set_v1.lower_bound(last_v0_key)) << '\n';
        std::cout << "offset_before: " << offset_before - range_start_offset << '\n';
        std::cout << "offset_before: " << offset_before << '\n';
        std::cout << "range_start_offset: " << range_start_offset << '\n';
        std::string message =
            std::string("query_range ") + std::to_string(i) + std::string(": [") +
            std::to_string(query_lower_bound) + std::string(", ") +
            std::to_string(query_upper_bound) + std::string(") ") +
            std::string(", expected an empty pair, found: {") + std::to_string(last_pair.first) +
            std::string(", ") + std::to_string(last_pair.second) + std::string("} ") +
            std::string(". At offset: ") + std::to_string(result_offset - range_start_offset) +
            std::string(" / ") + std::to_string(average_range_length);
        std::cout << message << std::endl;
      }
    }
    if (num_errors == max_errors) break;
  }

  std::cout << "Ok\nFound " << v1_found_keys.size() << " keys from v1";
  std::cout << std::endl;
  return num_errors;
}
}  // namespace utils
