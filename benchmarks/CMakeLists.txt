project (benchmarks)

# find_package(OpenMP)

add_executable(insert_find_bench insert_find_bench.cu)
set_target_properties(insert_find_bench PROPERTIES
								FOLDER "benchmarks")
target_link_libraries(insert_find_bench PRIVATE gpu_btrees)


add_executable(insert_find_blink_bench insert_find_blink_bench.cu)
set_target_properties(insert_find_blink_bench PROPERTIES
								FOLDER "benchmarks")
target_link_libraries(insert_find_blink_bench PRIVATE gpu_btrees )


add_executable(concurrent_insert_range_bench concurrent_insert_range_bench.cu)
set_target_properties(concurrent_insert_range_bench PROPERTIES
								FOLDER "benchmarks")
# target_compile_options(concurrent_insert_range_bench PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(concurrent_insert_range_bench PRIVATE gpu_btrees)

add_executable(concurrent_insert_range_variable_range_bench concurrent_insert_range_variable_range_bench.cu)
set_target_properties(concurrent_insert_range_variable_range_bench PROPERTIES
								FOLDER "benchmarks")
target_link_libraries(concurrent_insert_range_variable_range_bench PRIVATE gpu_btrees)

add_executable(concurrent_erase_find_bench concurrent_erase_find_bench.cu)
set_target_properties(concurrent_erase_find_bench PROPERTIES
								FOLDER "benchmarks")
target_link_libraries(concurrent_erase_find_bench PRIVATE gpu_btrees)

add_executable(concurrent_insert_range_reclaimer_bench concurrent_insert_range_reclaimer_bench.cu)
set_target_properties(concurrent_insert_range_reclaimer_bench PROPERTIES
								FOLDER "benchmarks")
target_link_libraries(concurrent_insert_range_reclaimer_bench PRIVATE gpu_btrees)
target_compile_definitions(concurrent_insert_range_reclaimer_bench
							PRIVATE COLLECT_RECLAIMER_STATS
							PRIVATE COLLECT_ALLOCATOR_STATS)


#find_versioned_test
# add_executable(find_versioned_bench find_versioned_bench.cu)
# set_target_properties(find_versioned_bench PROPERTIES
# 								FOLDER "benchmarks")
# target_compile_definitions(find_versioned_bench PRIVATE JSON_OUTPUT_DIR="${json_output_directory}")
# target_link_libraries(find_versioned_bench PRIVATE gpu_btrees)
