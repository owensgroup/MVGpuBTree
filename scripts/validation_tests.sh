#!/bin/bash
cd build

# This script will take ~6.5 hours

date >> validation_log.txt
./bin/concurrent_erase_find_bench --initial-size=45000000 --update-ratio=0.05 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_erase_find_bench --initial-size=45000000 --update-ratio=0.50 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_erase_find_bench --initial-size=45000000 --update-ratio=0.90 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt


##
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=32 --initial-size=40000000 --update-ratio=0.05 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=32 --initial-size=40000000 --update-ratio=0.50 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=32 --initial-size=40000000 --update-ratio=0.90 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt

##
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=32 --initial-size=1000000 --update-ratio=0.05 --num-ops=1000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=32 --initial-size=1000000 --update-ratio=0.50 --num-ops=1000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=32 --initial-size=1000000 --update-ratio=0.90 --num-ops=1000000 --validate-result=true --num-experiments=3 >> validation_log.txt

##
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=16 --initial-size=40000000 --update-ratio=0.05 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=16 --initial-size=40000000 --update-ratio=0.50 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=16 --initial-size=40000000 --update-ratio=0.90 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt

##
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=16 --initial-size=1000000 --update-ratio=0.05 --num-ops=1000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=16 --initial-size=1000000 --update-ratio=0.50 --num-ops=1000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=16 --initial-size=1000000 --update-ratio=0.90 --num-ops=1000000 --validate-result=true --num-experiments=3 >> validation_log.txt


##
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=8 --initial-size=40000000 --update-ratio=0.05 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=8 --initial-size=40000000 --update-ratio=0.50 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=8 --initial-size=40000000 --update-ratio=0.90 --num-ops=45000000 --validate-result=true --num-experiments=3 >> validation_log.txt

##
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=8 --initial-size=1000000 --update-ratio=0.05 --num-ops=1000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=8 --initial-size=1000000 --update-ratio=0.50 --num-ops=1000000 --validate-result=true --num-experiments=3 >> validation_log.txt
date >> validation_log.txt
./bin/concurrent_insert_range_bench --range-length=8 --initial-size=1000000 --update-ratio=0.90 --num-ops=1000000 --validate-result=true --num-experiments=3 >> validation_log.txt

date >> validation_log.txt

cat validation_log.txt | grep expected
if grep -q expected validation_log.txt; then
    echo "Found errors. Check the build/validation_log.txt."
else
    echo "All tests succeeded."
fi

cd ..