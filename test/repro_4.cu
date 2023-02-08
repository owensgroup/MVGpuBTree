#include <gpu_btree.h>
#include <cstdint>
#include <random>
#include <unordered_set>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <typename key_type, typename size_type, typename btree>
__global__ void modified_insert_kernel(const key_type* keys,
                                       const size_type keys_count,
                                       btree tree) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block     = cg::this_thread_block();
  auto tile      = cg::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key       = btree::invalid_key;
  auto value     = btree::invalid_value;
  bool to_insert = false;
  if (thread_id < keys_count) {
    key       = keys[thread_id];
    value     = thread_id;
    to_insert = true;
  }
  using allocator_type = typename btree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  size_type num_inserted = 1;
  auto work_queue        = tile.ballot(to_insert);
  while (work_queue) {
    auto cur_rank  = __ffs(work_queue) - 1;
    auto cur_key   = tile.shfl(key, cur_rank);
    auto cur_value = tile.shfl(value, cur_rank);

    tree.cooperative_insert(cur_key, cur_value, tile, allocator);

    if (tile.thread_rank() == cur_rank) { to_insert = false; }
    num_inserted++;
    work_queue = tile.ballot(to_insert);
  }
}

void investigate_tree_deadlock() {
  using key_type   = uint32_t;
  using value_type = uint32_t;

  size_t build_size       = size_t{1} << 25;
  key_type min_usable_key = 1;
  key_type max_usable_key = std::numeric_limits<key_type>::max() - 2;

  std::mt19937_64 gen(42);
  std::uniform_int_distribution<key_type> key_dist(min_usable_key, max_usable_key);
  std::vector<key_type> build_keys(build_size);
  std::unordered_set<key_type> build_keys_set;
  std::cout << "Generating keys.." << std::endl;

  while (build_keys_set.size() < build_size) {
    key_type key = key_dist(gen);
    build_keys_set.insert(key);
  }
  std::copy(build_keys_set.begin(), build_keys_set.end(), build_keys.begin());
  std::sort(build_keys.begin(), build_keys.end());

  key_type* keys_on_gpu;
  cudaMalloc(&keys_on_gpu, build_size * sizeof(key_type));
  cudaMemcpy(keys_on_gpu, build_keys.data(), build_size * sizeof(key_type), cudaMemcpyHostToDevice);
  for (size_t i = 0; i < 10000; ++i) {
    std::cout << "round " << i << " starting" << std::endl;

    GpuBTree::gpu_blink_tree<key_type, value_type, 16> tree;
    cuda_try(cudaPeekAtLastError());
    modified_insert_kernel<<<(build_size + 511) / 512, 512>>>(keys_on_gpu, build_size, tree);
    cuda_try(cudaPeekAtLastError());
    std::cout << "tree uses " << tree.compute_memory_usage() << " GB" << std::endl;
    cuda_try(cudaPeekAtLastError());
    std::cout << "round " << i << " done" << std::endl;
  }

  cudaFree(keys_on_gpu);
}

int main() {
  investigate_tree_deadlock();
  return 0;
}