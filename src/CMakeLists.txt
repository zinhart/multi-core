if(BuildCuda)
  set(concurrent_routines_lib
	  concurrent_routines.cu
	  concurrent_routines.cc
	  concurrent_routines_error.cc
	  serial/serial.cc
	  #parallel/parallel.cc
	  parallel/thread_pool.cc
  )
  # link with cuda lib
  cuda_add_library(concurrent_routines ${LIB_TYPE} ${concurrent_routines_lib})
  
  #link cpu threading library
  target_link_libraries(
   concurrent_routines 
   ${CUDA_CUBLAS_LIBRARIES}
   ${CMAKE_THREAD_LIBS_INIT}
  )

  if(BuildShared)
	#install  libconcurrent_routines.so to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS concurrent_routines LIBRARY DESTINATION lib LIBRARY DESTINATION lib)
  else()
	# install  libconcurrent_routines.a to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS concurrent_routines LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  endif()

  #to do
  install(DIRECTORY ../include/concurrent_routines DESTINATION include)
else()
  set(multi_core_lib 
	  multi_core.cc
	  serial/serial.cc
	  parallel/thread_pool.cc
	  parallel/priority_thread_pool.cc
     )	
   add_library(multi_core ${LIB_TYPE} ${multi_core_lib})
   
   #link cpu threading library & MKL
   target_link_libraries(
   multi_core 
   #   ${MKL_LIBRARIES}
   ${CMAKE_THREAD_LIBS_INIT}
  )

  if(BuildShared)
	#install  libmulti_core.so to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS multi_core LIBRARY DESTINATION lib DESTINATION lib)
  else()
	# install  libmulti_core.a to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS multi_core LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  endif()

  #to do
  install(DIRECTORY ../include/multi_core DESTINATION include)
endif()


