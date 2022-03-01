
min_range=4
max_range=1024
range_step=4

num_experiments=20

exec_dir="./bin"
exec_name="concurrent_insert_range_variable_range_bench"
device=0
output_dir="../results"
mkdir -p output_dir
additional_args="--validate-result=false --num-experiments=20"

update_ratios=(0.50)
initial_sizes=(100000 1'000'100 40'000'000)
num_concurrent_ops=(5'000'000)
for num_ops in "${num_concurrent_ops[@]}"
do
    for isize in "${initial_sizes[@]}"
    do
        for uratio in "${update_ratios[@]}"
        do
            for range in $(seq $min_range $range_step $max_range)
            do
                echo ${exec_dir}/${exec_name} --range-length=${range} --initial-size=${isize} --update-ratio=${uratio} --num-ops=${num_ops} --output-dir=${output_dir} ${additional_args}
                ${exec_dir}/${exec_name} --range-length=${range} --initial-size=${isize} --update-ratio=${uratio} --num-ops=${num_ops} --output-dir=${output_dir} ${additional_args}
            done
        done
    done
done