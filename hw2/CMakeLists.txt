#includes
include_directories(.)

#add mandelbrot as a library
cuda_add_library(mandelbrot ./mandelbrot/hw2.cu)

#link mandelbrot
target_link_libraries(mandelbrot ${CMAKE_THREAD_LIBS_INIT})

# cpu & gpu unit tests
if(BuildCuda)
  #hw src
  set(hw2_src hw2.cc)
add_executable(hw2 ${hw2_src})

target_link_libraries(hw2
					  mandelbrot
					  concurrent_routines
					  ${CMAKE_THREAD_LIBS_INIT}
					  )
add_test(
  NAME hw2
  COMMAND hw2
  )
endif()

