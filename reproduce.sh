#!/bin/bash
mkdir -p results

cd build

source ../scripts/bench_blink_vs_versioned.sh | tee -a ../results/bench_blink_vs_versioned.log
source ../scripts/bench_concurrent_insert_range.sh | tee -a ../results/bench_concurrent_insert_range.log
source ../scripts/bench_concurrent_erase_find.sh | tee -a ../results/bench_concurrent_erase_find.log
source ../scripts/bench_concurrent_insert_range_memory.sh | tee -a ../results/bench_concurrent_insert_range_memory.log
source ../scripts/bench_concurrent_insert_range_vary_range.sh | tee -a ../results/bench_concurrent_insert_range_vary_range.log

cd ..

#srun -p wario --gpus=1 reproduce.sh
#srun -p bowser --gpus=1 reproduce.sh
