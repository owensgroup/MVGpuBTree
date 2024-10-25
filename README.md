# [Fully Concurrent GPU Multiversion B-Tree](https://dl.acm.org/doi/10.1145/3559009.3569681)


<table><tr>
<th><b><a href="https://github.com/owensgroup/MVGpuBTree/tree/main/test">Examples/Tests</a></b></th>
<th><b><a href="https://github.com/owensgroup/MVGpuBTree/tree/main/benchmarks">Benchmarks</a></b></th>
<th><b><a href="apis.md">APIs</a></b></th>
<th><b><a href="reproduce.md">Reproduce</a></b></th>
</tr></table>


![Multiversion B-Tree](/docs/vbtree-nobg.png)


A fully concurrent GPU B-Tree that supports versioning (snapshots) and linearizable multipoint queries. Using our data structure and the tools we provide, you can launch one (or more) kernels where inside each kernel, you concurrently perform queries (e.g., point or range queries) and mutations (e.g., insert or update).


For more information, please check our PACT 2022 paper:

[**A GPU Multiversion B-Tree**](https://dl.acm.org/doi/10.1145/3559009.3569681)<br>
*[Muhammad A. Awad](https://maawad.github.io/), [Serban D. Porumbescu](https://web.cs.ucdavis.edu/~porumbes/), and [John D. Owens](https://www.ece.ucdavis.edu/~jowens/)*

The repository also contains:
1. [An implementation of our epoch-based memory reclamation strategy](https://github.com/owensgroup/MVGpuBTree/blob/main/include/memory_reclaimer.hpp)
2. [SlabAlloc memory allocator redesigned to allow building more complex allocators via composition](https://github.com/owensgroup/MVGpuBTree/blob/main/include/slab_alloc.hpp)
3. [Improved implementation of our B-Tree (reference B-Tree that doesn't support snapshots)](https://github.com/owensgroup/MVGpuBTree/blob/main/include/gpu_blink_tree.hpp)[^1].

### Our vision

GPU data structures such as the multiversion GPU B-Tree and other data structures we developed[^1][^2] should facilitate using them in the following concise and elegant manner:

```c++
#include<gpu_versioned_blink_tree.hpp>
#include<thrust/device_vector.hpp>
#include<thrust/for_each.hpp>

int main(){

 using key_t = uint32_t; using value_t = uint32_t;
 using tree_t = GpuBTree::gpu_versioned_blink_tree<key_t, value_t>;
 
 tree_t vtree(....); // call the data structure constructor 
 thrust::device_vector<key_t> keys(....); // initialize keys
 
 // solve a problem and do concurrent operations in a fully concurrent manner
 thrust::for_each(keys.begin(), keys.end(), [vtree](auto key){ 
  // perform operations in a tile-synchronous way
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tree_t::branching_factor>(block);
  // ... problem-specific code
  auto value = ...;
  vtree.cooperative_insert(key, value, tile, ...); // insert
  // ... maybe more problem-specific application code
  auto snapshot_id = vtree.take_snapshot(); // take snapshot
  // ... maybe even more problem-specific code
  auto found_value = vtree.cooperative_find(key, tile, snapshot_id, ...); // query
  // ... maybe even more problem-specific code
 });
}
```

The previous example illustrates our vision for using GPU data structures. To a large extent, we can do most of these operations using current CUDA/C++ abstractions and compilers; however, some of the APIs, such as memory allocators and reclaimers (especially on-device ones), still lack adequate support and standardization. BGHT[^2] provides the same device-side APIs and will require almost zero modifications to run the example snippet above.



## Requirements and limitations
Please create an issue if you face challenges with any of the following limitations and requirements.
### Requirements
* C++17/CUDA C++17
* NVIDIA Volta GPU or later microarchitectures
* CMake 3.18 or later
* CUDA 12.0 or later
* GPU with 20 GiBs or higher to run the benchmarks

### Limitations
* Keys and values must have a type of unsigned 32-bit
* Snapshot are limited to a maximum of 2^32 - 1 (can be extended to 2^64-1 easily)

## Reproducing the paper results
To reproduce the results, follow the following [steps](reproduce.md). Our [PACT 2022 paper](https://dl.acm.org/doi/10.1145/3559009.3569681) was awarded the [Results Reproduced v1.1](https://www.acm.org/publications/policies/artifact-review-and-badging-current) badge. If you find any mismatch (either faster or slower) between the results in the paper, please create an issue, and we will investigate the performance changes.

## Questions and bug report
Please create an issue. We will welcome any contributions that improve the usability and quality of our repository.

## BibTeX

The code in this repository is based on our Multiversion GPU B-Tree and GPU B-Tree publications:

```bibtex
@InProceedings{   Awad:2022:AGM,
  author        = {Muhammad A. Awad and Serban D. Porumbescu and John D.
                  Owens},
  title         = {A {GPU} Multiversion {B}-Tree},
  booktitle     = {Proceedings of the International Conference on Parallel
                  Architectures and Compilation Techniques},
  series        = {PACT 2022},
  year          = 2022,
  month         = oct,
  code          = {https://github.com/owensgroup/MVGpuBTree},
  doi           = {10.1145/3559009.3569681},
  url           = {https://escholarship.org/uc/item/4mz5t5b7},
  ucdcite       = {a146}
}
```
```bibtex
@InProceedings{   Awad:2019:EAH,
  author        = {Muhammad A. Awad and Saman Ashkiani and Rob Johnson and
                  Mart\'{\i}n Farach-Colton and John D. Owens},
  title         = {Engineering a High-Performance {GPU} {B}-Tree},
  booktitle     = {Proceedings of the 24th ACM SIGPLAN Symposium on
                  Principles and Practice of Parallel Programming},
  series        = {PPoPP 2019},
  year          = 2019,
  month         = feb,
  pages         = {145--157},
  acceptance    = {29 of 152 submissions, 19.1\%},
  doi           = {10.1145/3293883.3295706},
  acmauthorize  = {https://dl.acm.org/doi/10.1145/3293883.3295706?cid=81100458295},
  url           = {https://escholarship.org/uc/item/1ph2x5td},
  code          = {https://github.com/owensgroup/GpuBTree},
  ucdcite       = {a127}
}
```

[^1]: [Awad et al., Engineering a high-performance GPU B-Tree](https://escholarship.org/uc/item/1ph2x5td), https://github.com/owensgroup/GpuBTree
[^2]: [Awad et al., Analyzing and Implementing GPU Hash Tables](https://escholarship.org/uc/item/6cb1q6rz), https://github.com/owensgroup/BGHT
