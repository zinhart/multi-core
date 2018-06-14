if(BuildCuda)
  #includes
  include_directories(.)
	# cpu & gpu unit tests
	if(BuildCuda)
		#hw src
		set(hw4_src hw4.cc)
	add_executable(hw4 ${hw4_src})

	target_link_libraries(hw3
							concurrent_routines
							${CMAKE_THREAD_LIBS_INIT}
							)
	add_test(
		NAME hw4
		COMMAND hw4
		)
	endif()
endif()

