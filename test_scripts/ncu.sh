#ncu --target-processes application-only --replay-mode kernel --section regex:.* --cache-control all --apply-rules yes  --profile-from-start no
#--page raw --csv  ./bin/profiler_metrics &> test.csv
ncu --target-processes application-only --replay-mode kernel --section regex:.* --cache-control all --apply-rules yes  --profile-from-start no ./bin/find_versioned_test  --num-keys=10'000'000 --batch-size=10'000'000
ncu -o versioning_test --target-processes application-only --replay-mode kernel --section regex:.* --cache-control all --apply-rules yes  --profile-from-start no ./bin/find_versioned_test  --num-keys=10'000'000 --batch-size=10'000'000

ncu -o concurrent_insert_rq -f --target-processes application-only --replay-mode application --section regex:.* --cache-control all --apply-rules yes  --profile-from-start no ./bin/concurrent_insert_range_bench
ncu -o concurrent_erase_find_bench -f --target-processes application-only --replay-mode application --section regex:.* --cache-control all --apply-rules yes  --profile-from-start no ./bin/concurrent_insert_range_bench