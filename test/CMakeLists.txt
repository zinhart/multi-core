# includes
include_directories(.)

# add google test library
add_library(gtest_concurrent_routines ./gtest/gtest-all.cc)

# link google test
target_link_libraries(gtest_concurrent_routines ${CMAKE_THREAD_LIBS_INIT})

# cpu & gpu unit tests
if(BuildCuda)
  #unit test src
  set(unit_tests_src
   run_all.cc
   thread_pool_test.cc
   # cpu_test.cc
   gpu_test.cc 
   )
add_executable(unit_tests ${unit_tests_src})

target_link_libraries(unit_tests 
					  gtest_concurrent_routines
					  concurrent_routines
					  ${CMAKE_THREAD_LIBS_INIT}
					  )
add_test(
  NAME unit_tests
  COMMAND unit_tests
  )
# cpu only unit tests
else()
  #unit test src
  set(unit_tests_src
   run_all.cc
   thread_pool_test.cc
   cpu_test.cc
   )
add_executable(unit_tests ${unit_tests_src})

target_link_libraries(unit_tests 
					  gtest_concurrent_routines
					  concurrent_routines
				   	  ${MKL_LIBRARIES}
					  ${CMAKE_THREAD_LIBS_INIT}
					  )
add_test(
  NAME unit_tests
  COMMAND unit_tests
  )
endif()

