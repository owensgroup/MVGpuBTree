#srun -p daisy --gpus=1 ./run.sh
# srun -p bowser --gpus=V100:1 ./reproduce_bowser.sh
# srun -p daisy --gpus=1 ./reproduce_daisy.sh
srun -p bowser --gpus=V100:1 ./scripts/bench_concurrent_insert_range.sh
