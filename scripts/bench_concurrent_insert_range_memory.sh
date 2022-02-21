
num_ops=(45'000'000)

exec_dir="./bin"
exec_name="concurrent_insert_range_reclaimer_bench"
device=0
output_dir="../results"
mkdir -p output_dir
additional_args="--validate=false"

update_ratios=(0.50)
range_length=(16)
initial_sizes=45'000'000

for range in "${range_length[@]}"
do
    for uratio in "${update_ratios[@]}"
    do
        for nop in "${num_ops[@]}"
        do
            echo ${exec_dir}/${exec_name} --range-length=${range} --initial-size=${initial_sizes} --update-ratio=${uratio} --num-ops=${nop} --output-dir=${output_dir} ${additional_args}
            ${exec_dir}/${exec_name} --range-length=${range} --initial-size=${initial_sizes} --update-ratio=${uratio} --num-ops=${nop} --output-dir=${output_dir} ${additional_args}
        done
    done
done