# run from current directory
mkdir -p figs


for dir in ../results/*/
do
    dir=${dir%*/}
    GPU_NAME=${dir##*/}

    echo "Plotting results for GPU with name: $GPU_NAME"

    input_dir="../results/"$GPU_NAME
    output_dir="./figs/"$GPU_NAME
    mkdir -p $output_dir

    min_find=-1
    max_find=-1

    min_rate=-1
    max_rate=-1

    min_erase_find=-1
    max_erase_find=-1

    # uncomment below to use same axes minimum and maximum as in the paper
    # min_find=1300
    # max_find=2200

    # min_rate=0
    # max_rate=300

    # min_erase_find=200
    # max_erase_find=500

    # Figure 2
    exec_name=plot_blink_vs_versioned
    python ./${exec_name}.py -d ${input_dir} -od $output_dir -mf ${min_find} -xf ${max_find} -mi ${min_rate} -xi ${max_rate}&> ${output_dir}/${exec_name:5}.txt

    # Figure 3 and Table 1
    exec_name=plot_concurrent_insert_range
    python ./${exec_name}.py -d ${input_dir} -od $output_dir -mf ${min_rate} -xf ${max_rate}&> ${output_dir}/${exec_name:5}.txt

    # Figure 4
    exec_name=plot_concurrent_insert_range_vary_range
    python ./${exec_name}.py -d ${input_dir} -od $output_dir &> ${output_dir}/${exec_name:5}.txt

    # Figure 5 and Table 2
    exec_name=plot_concurrent_erase_find
    python ./${exec_name}.py -d ${input_dir} -od $output_dir -mf $min_erase_find -xf $max_erase_find&> ${output_dir}/${exec_name:5}.txt

    # Figure 6
    exec_name=plot_concurrent_insert_range_reclamation_stats
    python ./${exec_name}.py -d ${input_dir} -od $output_dir &> ${output_dir}/${exec_name:5}.txt

done