min_num_keys=1'000'000
max_num_keys=70'000'000
num_keys_step=1'000'000
num_experiments=5

exec_dir="./bin"
exec_name="insert_find_blink_bench"
device=0
output_dir="../results"
mkdir -p output_dir
additional_args="--validate-result=false --validate-tree=false --exist-ratio=1.0"

for num_keys in $(seq $min_num_keys $num_keys_step $max_num_keys)
do
    echo ${exec_dir}/${exec_name} --num-keys=${num_keys} --num-experiments=${num_experiments} --output-dir=${output_dir} ${additional_args}
    ${exec_dir}/${exec_name} --num-keys=${num_keys} --num-experiments=${num_experiments} --output-dir=${output_dir} ${additional_args}
done