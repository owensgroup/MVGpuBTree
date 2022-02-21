add_library(gpu_btrees_src INTERFACE)

file(GLOB_RECURSE   gpu_btrees_src
                    "*.h"
                    "*.hpp"
                    "*.cuh"
                    "*.cu")
set(SOURCE_LIST ${gpu_btrees_src})
target_sources(gpu_btrees INTERFACE ${gpu_btrees_src})
target_include_directories(gpu_btrees INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")