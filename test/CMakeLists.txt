project (test)


# bulk build
add_executable(test_bulk_build test_bulk_build.cu)
set_target_properties(test_bulk_build PROPERTIES
								FOLDER "tests")
target_link_libraries(test_bulk_build PRIVATE gpu_btrees)



# concurrent insert range query
add_executable(concurrent_insert_rq_vtree concurrent_insert_rq_vtree.cu)
set_target_properties(concurrent_insert_rq_vtree PROPERTIES
								FOLDER "tests")
target_link_libraries(concurrent_insert_rq_vtree PRIVATE gpu_btrees)


#erase_test
add_executable(erase_test erase_test.cu)
set_target_properties(erase_test PROPERTIES
								FOLDER "tests")
target_link_libraries(erase_test PRIVATE gpu_btrees)



#find_blink_test
add_executable(find_blink_test find_blink_test.cu)
set_target_properties(find_blink_test PROPERTIES FOLDER "tests")
target_link_libraries(find_blink_test PRIVATE gpu_btrees)



#find_versioned_test
add_executable(find_versioned_test find_versioned_test.cu)
set_target_properties(find_versioned_test PROPERTIES FOLDER "tests")
target_link_libraries(find_versioned_test PRIVATE gpu_btrees)


# test_slab_alloc
add_executable(test_slab_alloc test_slab_alloc.cu)
set_target_properties(test_slab_alloc PROPERTIES
								FOLDER "tests")
target_link_libraries(test_slab_alloc PRIVATE gpu_btrees)



# unit tests
include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.11.0
  VERSION 1.11.0
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)

add_executable(unittest_btree unittest_btree.cu)
target_link_libraries(unittest_btree	PRIVATE gpu_btrees  PRIVATE gtest)

add_executable(unittest_versioning unittest_versioning.cu)
target_link_libraries(unittest_versioning PRIVATE gpu_btrees  PRIVATE gtest)


