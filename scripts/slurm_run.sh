#srun -p daisy --gpus=1 ./run.sh
# srun -p bowser --gpus=V100:1 ./reproduce_bowser.sh
# srun -p daisy --gpus=1 ./reproduce_daisy.sh
srun -p bowser --gpus=V100:1 ./validation_tests.sh
