if(BuildCuda)
  #includes
  include_directories(.)
	# cpu & gpu unit tests
	if(BuildCuda)
		#hw src
		set(hw3_src hw3.cc)
	add_executable(hw3 ${hw3_src})

	target_link_libraries(hw3
							concurrent_routines
							${CMAKE_THREAD_LIBS_INIT}
							)
	add_test(
		NAME hw3
		COMMAND hw3
		)
	endif()
endif()

