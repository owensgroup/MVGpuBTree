# GPU Multiversion B-Tree


<table><tr>
<th><b><a href="https://github.com/owensgroup/MVGpuBTree/tree/main/test">Examples/Tests</a></b></th>
<th><b><a href="https://github.com/owensgroup/MVGpuBTree/tree/main/benchmarks">Benchmarks</a></b></th>
<th><b><a href="apis.md">APIs</a></b></th>
<th><b><a href="reproduce.md">Reproduce</a></b></th>
</tr></table>

A GPU B-Tree that supports versioning (snapshots) and linearizable multipoint queries.
For more information, please check our paper:

[**A GPU Multiversion B-Tree**](https://escholarship.org/uc/item/5kc834wm#page=52)<br>
*[Muhammad A. Awad](https://maawad.github.io/), [Serban D. Porumbescu](https://web.cs.ucdavis.edu/~porumbes/), and [John D. Owens](https://www.ece.ucdavis.edu/~jowens/)*

The repository also contains:
1. [An implementation of our epoch-based memory reclamation strategy](https://github.com/owensgroup/MVGpuBTree/blob/main/include/memory_reclaimer.hpp)
2. [SlabAlloc memory allocator redesigned to allow building more complex allocators via composition](https://github.com/owensgroup/MVGpuBTree/blob/main/include/slab_alloc.hpp)
3. [Improved implementation of our B-Tree (reference B-Tree that doesn't support snapshots)](https://github.com/owensgroup/MVGpuBTree/blob/main/include/gpu_blink_tree.hpp).



## Requirements and limitations
Please create an issue if you face challenges with any of the following limitations and requirements.
### Requirements
* C++17/CUDA C++17
* NVIDIA Volta GPU or later microarchitectures
* CMake 3.18 or later
* CUDA 11.5 or later
* GPU with 20 GiBs or higher to run the benchmarks

### Limitations
* Keys and values must have a type of unsigned 32-bit
* Snapshot are limited to a maximum of 2^32 - 1 (can be extended to 2^64-1 easily)

## Reproducing the paper results
To reproduce the results, follow the following [steps](reproduce.md). If you find any mismatch (either faster or slower) between the results in the paper, please create an issue, and we will investigate the performance changes.

## Questions and bug report
Please create an issue. We will welcome any contributions that improve the usability and quality of our repository.

## BibTeX
```bibtex
@inproceedings{Awad:2022:AGM,
  author = {Muhammad A. Awad and Serban D. Porumbescu and John D. Owens},
  title = {A {GPU} Multiversion {B}-Tree},
  booktitle = {Proceedings of the International Conference on Parallel
              Architectures and Compilation Techniques},
  series = {PACT 2022},
  year = 2022,
  month = oct,
  code = {https://github.com/owensgroup/MVGpuBTree},
  doi = {10.1145/3559009.3569681}
 }
```
