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
#include <limits>

template <typename Key, typename Value>
struct __align__(8) pair_type {
  using key_type   = Key;
  using value_type = Value;
  using size_type  = uint32_t;
  HOST_DEVICE_QUALIFIER pair_type(const key_type& key, const value_type& value)
      : first(key), second(value) {}
  HOST_DEVICE_QUALIFIER pair_type(void)
      : first(std::numeric_limits<key_type>::max())
      , second(std::numeric_limits<value_type>::max()){};
  HOST_DEVICE_QUALIFIER pair_type(const pair_type<key_type, value_type>& other) {
    first  = other.first;
    second = other.second;
  }

  HOST_DEVICE_QUALIFIER pair_type(const volatile pair_type<key_type, value_type>& other) {
    first  = other.first;
    second = other.second;
  }
  HOST_DEVICE_QUALIFIER volatile pair_type<key_type, value_type>& operator=(
      const volatile pair_type<key_type, value_type>& other) volatile {
    first  = other.first;
    second = other.second;
    return *this;
  }
  HOST_DEVICE_QUALIFIER pair_type<key_type, value_type>& operator=(
      const pair_type<key_type, value_type>& other) {
    first  = other.first;
    second = other.second;
    return *this;
  }

  HOST_DEVICE_QUALIFIER bool operator==(const pair_type<key_type, value_type>& other) {
    return (first == other.first) && (second == other.second);
  }
  HOST_DEVICE_QUALIFIER bool operator!=(const pair_type<key_type, value_type>& other) {
    return (first != other.first) || (second != other.second);
  }

  key_type first;
  value_type second;

 private:
};
