
min_num_ops=1'000'000
max_num_ops=45'000'000
num_ops_step=1'000'000

num_experiments=20

exec_dir="./bin"
exec_name="concurrent_erase_find_bench"
device=0
output_dir="../results"
mkdir -p output_dir
additional_args="--validate=false --num-experiments=20"

update_ratios=(0.05 0.50 0.90)
initial_sizes=(45'000'000)
for isize in "${initial_sizes[@]}"
do
    for uratio in "${update_ratios[@]}"
    do
        for num_ops in $(seq $min_num_ops $num_ops_step $max_num_ops)
        do
            echo ${exec_dir}/${exec_name} --initial-size=${isize} --update-ratio=${uratio} --num-ops=${num_ops} --output-dir=${output_dir} ${additional_args}
            ${exec_dir}/${exec_name} --initial-size=${isize} --update-ratio=${uratio} --num-ops=${num_ops} --output-dir=${output_dir} ${additional_args}
        done
    done
done