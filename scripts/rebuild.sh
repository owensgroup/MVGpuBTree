#!/bin/bash

build_dir="build"
targets=("all")

cmake -B $build_dir
cmake --build $build_dir --target "${targets[@]}" --parallel $(nproc)