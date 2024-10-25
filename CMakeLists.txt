cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()


project(GpuBTree LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)



# options
option(build_benchmarks "Build benchmarks" ON)
option(build_tests "Build tests" ON)


# Auto-detect GPU architecture, sets ${CUDA_ARCHS}
include("cmake/AutoDetectCudaArch.cmake")

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(json_output_directory "${CMAKE_CURRENT_SOURCE_DIR}/outputs")


# B-Tree lib
add_library(gpu_btrees INTERFACE)

# CUDA and C++ compiler flags
set(CXX_FLAGS
        $<IF:$<CXX_COMPILER_ID:MSVC>,
        /std:c++17,
        --std=c++17
        >)
set(CUDA_FLAGS
    $<IF:$<CXX_COMPILER_ID:MSVC>,
    -Xcompiler=/std:c++17,
    -Xcompiler=-std=c++17 -Wextra>
    $<IF:$<CXX_COMPILER_ID:MSVC>,
    -Xcompiler=/W4,
    -Xcompiler=-Wall -W  -Wcast-qual -Wuninitialized>
    $<IF:$<CONFIG:Debug>,
        -G -src-in-ptx,
        -lineinfo>
    ${CUDA_ARCHS}
    -maxrregcount=70
	--expt-extended-lambda
    --expt-relaxed-constexpr
    -Wno-deprecated-gpu-targets #silence deprecated architectures
    # $<$<CONFIG:Release>:-Xptxas -warn-spills -res-usage>
    -Xptxas -warn-spills
    -res-usage
    )


set(external_includes "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(rapidjson_includes "${rapidjson_SOURCE_DIR}/include")
set(btree_includes "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(developer_flags INTERFACE)
target_include_directories( developer_flags
                            INTERFACE ${external_includes}
                            INTERFACE ${rapidjson_includes})
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)

target_link_libraries(gpu_btrees
    INTERFACE $<BUILD_INTERFACE:developer_flags>
    INTERFACE $<IF:$<CXX_COMPILER_ID:MSVC>,,-lstdc++fs>)



# Visual studio setup
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
## Fix includes in visual studio for intellisense
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT insert_blink_custom_alloc)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT insert_versioned_test)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT find_versioned_test)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT find_versioned_bench)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT concurrent_blink_custom_alloc)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT concurrent_insert_rq_vtree)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT reclaimation_test)
#set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT test_slab_alloc)

file(TO_NATIVE_PATH ${external_includes} external_includes_native)
file(TO_NATIVE_PATH ${btree_includes} btree_includes_native)
file(TO_NATIVE_PATH ${rapidjson_includes} rapidjson_includes_native)

set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
    $(VC_IncludePath)
    $(WindowsSDK_IncludePath)
    ${external_includes_native}
    ${btree_includes_native}
    ${rapidjson_includes_native}
    )


add_subdirectory(include)
if(build_tests)
   add_subdirectory(test)
endif(build_tests)

if(build_benchmarks)
   add_subdirectory(benchmarks)
endif(build_benchmarks)
