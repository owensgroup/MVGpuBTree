# GPU Multiversion B-Tree


<table><tr>
<th><b><a href="https://github.com/owensgroup/MVGpuBTree/tree/main/test">Examples/Tests</a></b></th>
<th><b><a href="https://github.com/owensgroup/MVGpuBTree/tree/main/benchmarks">Benchmarks</a></b></th>
<th><b><a href="apis.md">APIs</a></b></th>
</tr></table>

A GPU B-Tree that supports versioning (snapshots) and linearizable multipoint queries.
For more information, please check our paper:

[**A GPU Multiversion B-Tree**]()<br>
*[Muhammad A. Awad](https://maawad.github.io/), [Serban D. Porumbescu](https://web.cs.ucdavis.edu/~porumbes/), and [John D. Owens](https://www.ece.ucdavis.edu/~jowens/)*




## Requirements and limitations
Please create an issue if you face challenges with any of the following limitations and requirements.
### Requirements
* C++17/CUDA C++17
* NVIDIA Volta GPU or later microarchitectures
* CMake 3.8 or later
* CUDA 11.5 or later
* GPU with 16 GiBs or higher to run the benchmarks

### Limitations
* Keys and values must have a type of unsigned 32-bit
* Snapshot are limited to a maximum of 2^32 - 1 (can be extended to 2^64-1 easily)

## Reproducing the paper results
To reproduce the results, follow the following [steps](reproduce.md). If you find any mismatch (either faster or slower) between the results in the paper, please create an issue, and we will investigate the performance changes.

## Questions and bug report
Please create an issue. We will welcome any contributions that improve the usability and quality of our repository.

## Bibtex
```
```