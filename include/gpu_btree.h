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
template <class T>
struct device_allocator_context {};
#include <macros.hpp>

// order matters
#include <pair_type.hpp>

namespace GpuBTree {
template <typename Key, typename Value, int b = 16>
struct node_type {
  using T = pair_type<Key, Value>;
  T node[b];
};
}  // namespace GpuBTree

#include "gpu_blink_tree.hpp"
#include "gpu_versioned_blink_tree.hpp"
