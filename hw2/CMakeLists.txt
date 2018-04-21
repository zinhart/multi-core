#includes
include_directories(.)

# cpu & gpu unit tests
if(BuildCuda)
  #unit test src
  set(hw2_src
	hw2.cc
   )
add_executable(hw2 ${hw2_src})

target_link_libraries(hw2
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
   cpu_test.cc
   )
add_executable(unit_tests ${unit_tests_src})

target_link_libraries(unit_tests 
					  gtest
					  concurrent_routines
					  ${CMAKE_THREAD_LIBS_INIT}
					  )
add_test(
  NAME unit_tests
  COMMAND unit_tests
  )
endif()

