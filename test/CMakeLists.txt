project (test)

set(TESTS
    test_bulk_build
    concurrent_insert_rq_vtree
    erase_test
    find_blink_test
    find_versioned_test
    test_slab_alloc
)

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cu)
    set_target_properties(${TEST} PROPERTIES FOLDER "tests")
    target_link_libraries(${TEST} PRIVATE gpu_btrees)
endforeach()


# unit tests
include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.15.2
  VERSION 1.15.2
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)

set(GTEST_TESTS
    unittest_btree
    unittest_versioning
)
foreach(TEST ${GTEST_TESTS})
    add_executable(${TEST} ${TEST}.cu)
    set_target_properties(${TEST} PROPERTIES FOLDER "tests")
    target_link_libraries(${TEST} PRIVATE gpu_btrees PRIVATE gtest)
endforeach()


