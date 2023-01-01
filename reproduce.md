# Reproducing the paper's results

## Archived Results
For reference, the repository contains archived results from our paper ([Figures](https://github.com/owensgroup/MVGpuBTree/tree/main/archived-figs/Tesla-V100-PCIE-32GB), [CSV files](https://github.com/owensgroup/MVGpuBTree/tree/main/archived-results/Tesla-V100-PCIE-32GB)).

## Building the code

### Requirements:
#### Hardware
* NVIDIA Volta GPU or later microarchitectures with at least 20 GiBs of DRAM and shared memory per block greater than 1556 bytes
* CPU DRAM usage will not exceed 1 GiB
* In addition to unit tests, additional extensive validation is available. However, it will increase the GPU DRAM requirements up to 30 GiBs and CPU DRAM usage up to 30 GiBs. See the Additional validation section for instructions on running and reducing the memory requirements.
#### Software
You can use our [docker image](https://hub.docker.com/repository/docker/maawad/mvgpubtree) or  a system with the following software:
* CUDA 11.5 or later
* C++17/CUDA C++17
* CMake 3.18 or later
* Linux OS

To plot the result, you need a python 3.9.7 or higher with the following libraries installed:
* `matplotlib`
* `pandas`
* `scipy`

Refer to [Python guides](https://docs.python-guide.org/starting/install3/linux/) for instructions on how to install Python and its packages.
### Building

```bash
git clone https://github.com/owensgroup/MVGpuBTree.git
cd MVGpuBTree
mkdir build && cd build
cmake ..
make -j
cd ..
```

### Unit tests
Run the unit tests to check unit tests validation:
```bash
./build/bin/unittest_btree
./build/bin/unittest_versioning
```

## Running all the benchmarks and plotting
Time:  ~7 hours on an Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz and a Tesla-V100-PCIE-32GB.
```bash
# Run the scripts from the top-level Git directory
# Running the experiments
source reproduce.sh         # Runs the benchmarks and stores the benchmark results in csv format
                            # The results will be stored under results/GPU-name (e.g., results/Tesla-V100-PCIE-32GB)
# Plotting the results
cd plots
source plot.sh              # The script will iterate over all GPU-names in results directory and plot them

cd figs/GPU-name            # Replace GPU name with the name of your GPU used in benchmarking
                            # Check the directory name where the results is stored if you are not sure about the GPU name
ls
# Figure 2
insertion_find_rates_slab.pdf           # Rates for insertion and point query for B-Tree vs. VBTree
insertion_find_rates_slab.svg           # Same as above in svg format
blink_vs_versioned.txt             # Tabular summary of the  B-Tree vs. VBTree results

# Figure 3 and Table 1
insertion_rq_rates_slab1.pdf            # Rates for operations (insert and RQ) on an initial tree size of 1 million keys
insertion_rq_rates_slab1.svg            # Same as above in svg format
insertion_rq_rates_slab40.pdf           # Rates for operations (insert and RQ) on an initial tree size of 40 million keys
insertion_rq_rates_slab40.svg           # Same as above in svg format
concurrent_insert_range.txt        # Tabular summary of the concurrent insert RQ results

# Figure 4
insertion_vary_rq_rates_initial1M_update50_num_ops5_slab.pdf    # Rates of operations using 1 million keys initial tree size and variable range length
insertion_vary_rq_rates_initial1M_update50_num_ops5_slab.svg    # Same as above in svg format
insertion_vary_rq_rates_initial40M_update50_num_ops5_slab.pdf   # Rates of operations using 50 million keys initial tree size and variable range length
insertion_vary_rq_rates_initial40M_update50_num_ops5_slab.svg   # Same as above in svg format

# Figure 5 and Table 2
erase_find_rates_slab45.pdf             # Rates for operations (find and erase) on an initial tree size of 45 million keys
erase_find_rates_slab45.svg             # Same as above in svg format
concurrent_erase_find.txt          # Tabular summary of the concurrent find erase results

# Figure 6
insertion_find_memory_45m_45m_50_16_slab.pdf            # Figure 5.a
insertion_find_memory_45m_45m_50_16_slab.svg            # Same as above in svg format
insertion_find_ratio_memory_45m_45m_50_16_slab.pdf      # Figure 5.b
insertion_find_ratio_memory_45m_45m_50_16_slab.svg      # Same as above in svg format
concurrent_insert_range_reclamation_stats.txt      # Empty log file


# Not included in the paper
insertion_vary_rq_rates_initial0M_update50_num_ops5_slab.pdf    # Rates of operations using 500 thousands keys initial tree size and variable range length
insertion_vary_rq_rates_initial0M_update50_num_ops5_slab.svg    # Same as above in svg format
```

## Running one benchmark at a time

### Generating benchmark results
Navigate to the build directory
```bash
cd build
```
#### B-Tree vs. versioned B-Tree (ViB-Tree and VOB-Tree) (Figure 2)
~30 mins [^1]
```bash
source ../scripts/bench_blink_vs_versioned.sh | tee -a ../results/bench_blink_vs_versioned.log
```
#### Concurrent insertion and range query benchmark (Figure 3 and Table 1)
~3 hours and 30 minutes [^1]
```bash
source ../scripts/bench_concurrent_insert_range.sh | tee -a ../results/bench_concurrent_insert_range.log
```

#### Concurrent insertion and variable range query benchmark (Figure 4)
~2 hours [^1]
```bash
source ../scripts/bench_concurrent_insert_range_vary_range.sh | tee -a ../results/bench_concurrent_insert_range_vary_range.log
```

#### Concurrent insertion and erase benchmark (Figure 5 and Table 2)
~ 1 hour [^1]
```bash
source ../scripts/bench_concurrent_erase_find.sh | tee -a ../results/bench_concurrent_erase_find.log
```

#### Memory reclamation benchmark (Figure 6)
~ 1 minute [^1]

```bash
source ../scripts/bench_concurrent_insert_range_memory.sh | tee -a ../results/bench_concurrent_insert_range_memory.log
```

[^1]: On a system with an Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz and a Tesla-V100-PCIE-32GB
### Plotting

First, navigate to the plots directory:
```bash
cd ../plots
```
Then, set some variables:
```bash
GPU_NAME="your_GPU_name"                    # GPU name (check the results folder if you are not sure about the GPU name)
input_dir="../results/"$GPU_NAME            # All executable generated results (in csv format) will be stored under this directory
output_dir="./figs/"$GPU_NAME               # Where we will store the figures
mkdir -p $output_dir                        # Creating results directory

## Setting minimum and maximum axis values.
# These variables are optional.
# If you would like the plotting script to set them automatically, set each variable to -1

min_rate=0                                  # Y-axis minimum for Figures 2 (insert rate) and 3
max_rate=300                                # Y-axis maximum for Figures 2 (insert rate) and 3

min_erase_find=200                          # Y-axis minimum for Figure 4
max_erase_find=500                          # Y-axis maximum for Figure 4

min_find=1300                               # Y-axis minimum for Figure 2 (find rate)
max_find=2200                               # Y-axis maximum for Figure 2 (find rate)
```

#### B-Tree vs. versioned B-Tree (ViB-Tree and VOB-Tree) (Figure 2)
```bash
exec_name=plot_blink_vs_versioned
python ./${exec_name}.py -d ${input_dir} -od $output_dir -mf ${min_find} -xf ${max_find} -mi ${min_rate} -xi ${max_rate}&> ${output_dir}/${exec_name}.txt

# results
cd $output_dir
ls
insertion_find_rates_slab.pdf           # Rates for insertion and point query for B-Tree vs. VBTree
insertion_find_rates_slab.svg           # Same as above in svg format
plot_blink_vs_versioned.txt             # Tabular summary of the  B-Tree vs. VBTree results
```
#### Concurrent insertion and range query benchmark (Figure 3 and Table 1)
```bash
exec_name=plot_concurrent_insert_range
python ./${exec_name}.py -d ${input_dir} -od $output_dir -mf ${min_rate} -xf ${max_rate}&> ${output_dir}/${exec_name}.txt

# results
cd $output_dir
ls
insertion_rq_rates_slab1.pdf            # Rates for operations on an initial tree size of 1 million keys
insertion_rq_rates_slab1.svg            # Same as above in svg format
insertion_rq_rates_slab40.pdf           # Rates for operations on an initial tree size of 40 million keys
insertion_rq_rates_slab40.svg           # Same as above in svg format
plot_concurrent_insert_range.txt        # Tabular summary of the concurrent insert rq results

```

#### Concurrent insertion and variable range query length benchmark (Figure 4)
```bash
exec_name=plot_concurrent_insert_range_vary_range
python ./${exec_name}.py -d ${input_dir} -od $output_dir &> ${output_dir}/${exec_name}.txt

# Results
cd $output_dir
ls
insertion_vary_rq_rates_initial1M_update50_num_ops5_slab.pdf    # Rates of operations using 1 million keys initial tree size and variable range length
insertion_vary_rq_rates_initial1M_update50_num_ops5_slab.svg    # Same as above in svg format
insertion_vary_rq_rates_initial40M_update50_num_ops5_slab.pdf   # Rates of operations using 50 million keys initial tree size and variable range length
insertion_vary_rq_rates_initial40M_update50_num_ops5_slab.svg   # Same as above in svg format
```

#### Concurrent insertion and erase benchmark (Figure 5 and Table 2)

```bash
exec_name=plot_concurrent_erase_find
python ./${exec_name}.py -d ${input_dir} -od $output_dir -mf $min_erase_find -xf $max_erase_find&> ${output_dir}/${exec_name}.txt

# Results
cd $output_dir
ls
erase_find_rates_slab45.pdf             # Rates for operations on an initial tree size of 45 million keys
erase_find_rates_slab45.svg             # Same as above in svg format
plot_concurrent_erase_find.txt          # Tabular summary of the concurrent find erase results
```

#### Memory reclamation benchmark (Figure 6)
```bash
exec_name=plot_concurrent_insert_range_reclamation_stats
python ./${exec_name}.py -d ${input_dir} -od $output_dir &> ${output_dir}/${exec_name}.txt

# Results
cd $output_dir
ls
insertion_find_memory_45m_45m_50_16_slab.pdf            # Figure 5.a
insertion_find_memory_45m_45m_50_16_slab.svg            # Same as above in svg format
insertion_find_ratio_memory_45m_45m_50_16_slab.pdf      # Figure 5.b
insertion_find_ratio_memory_45m_45m_50_16_slab.svg      # Same as above in svg format
plot_concurrent_insert_range_reclamation_stats.txt      # Empty log file
```

# Additional validation

In addition to the unit tests, we provide additional extensive validation. All scripts contain a flag to turn validation on. Setting `validate_results` or `validate` to true will validate the query results. However, validation is relatively slow (especially for range query benchmark) as it performs all operations serially on the CPU. Additional CPU memory will be required (~30 GiBs). GPU memory usage can go up to 30 GiBs. We recommend running single experiments and not the entire benchmarking script to test validation. Most memory usage is used for range query results; therefore, reducing the range query size will significantly reduce memory usage.


For convenience, a script that performs tests is provided. The script took ~6.5 hours on our system. From the `MVGpuBTre` top-level directory, run the following script:
```bash
source scripts/validation_tests.sh
```
A successful validation will return:
```
All tests succeeded.
```


Examples for how to perform one test at a time:
```bash
# Run concurrent insert range query benchmark with:
#   Range length of 32
#   Initial tree size 40'000'000 pairs
#   Update ratio 0.05
#   Number of concurrent operations  45'000'000 (divided using the update ratio)
#   Validate result is set to true
#   Number of experiments is 3 (validate each experiment)
# Takes 1 hour 20 mins on our system
./bin/concurrent_insert_range_bench --range-length=32 --initial-size=40000000 --update-ratio=0.05 --num-ops=45000000 --validate-result=true --num-experiments=3

# Run concurrent erase and point query benchmark with:
#   Initial tree size 45'000'000 pairs
#   Update ratio 0.9
#   Number of concurrent operations  45'000'000 (divided using the update ratio)
#   Validate result is set to true
#   Number of experiments is 3 (validate each experiment)
# Takes 2 minutes
./bin/concurrent_erase_find_bench --initial-size=45000000 --update-ratio=0.90 --num-ops=45000000 --validate-result=true --num-experiments=3
```
