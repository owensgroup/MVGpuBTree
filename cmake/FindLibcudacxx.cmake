cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
CPMAddPackage(
  NAME libcudacxx
  GITHUB_REPOSITORY NVIDIA/libcudacxx
  GIT_TAG main
  OPTIONS
     "build_tests OFF"
     "build_benchmarks OFF"
)


set(libcudacxx_includes "${libcudacxx_SOURCE_DIR}/include")
