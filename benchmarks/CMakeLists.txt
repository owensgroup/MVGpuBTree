project(benchmarks)

set(BENCHMARKS
    insert_find_bench
    insert_find_blink_bench
    concurrent_insert_range_bench
    concurrent_insert_range_variable_range_bench
    concurrent_erase_find_bench
)

foreach(BENCHMARK ${BENCHMARKS})
    add_executable(${BENCHMARK} ${BENCHMARK}.cu)
    set_target_properties(${BENCHMARK} PROPERTIES FOLDER "benchmarks")
    target_link_libraries(${BENCHMARK} PRIVATE gpu_btrees)
endforeach()

add_executable(concurrent_insert_range_reclaimer_bench concurrent_insert_range_reclaimer_bench.cu)
set_target_properties(concurrent_insert_range_reclaimer_bench PROPERTIES FOLDER "benchmarks")
target_link_libraries(concurrent_insert_range_reclaimer_bench PRIVATE gpu_btrees)
target_compile_definitions(concurrent_insert_range_reclaimer_bench
							PRIVATE COLLECT_RECLAIMER_STATS
							PRIVATE COLLECT_ALLOCATOR_STATS)