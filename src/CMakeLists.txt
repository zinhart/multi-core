if(BuildCuda)
  set(concurrent_routines_lib
	  gpu.cu
  )
  # link with cuda lib
  cuda_add_library(concurrent_routines ${LIB_TYPE} ${concurrent_routines_lib})
  # add cpu threads
  set(concurrent_routines_lib 
	${concurrent_routines_lib}
	cpu.cc
     )
   #link cpu threading library
   target_link_libraries(
   concurrent_routines 
   # ${concurrent_routines_lib}
   ${CMAKE_THREAD_LIBS_INIT}
  )

  if(BuildShared)
	#install  libconcurrent_routines.a to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS concurrent_routines LIBRARY DESTINATION lib LIBRARY DESTINATION lib)
  else()
	# install  libconcurrent_routines.a to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS concurrent_routines LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  endif()

  #to do
  install(DIRECTORY ../include/concurrent_routines DESTINATION include)
else()
  set(concurrent_routines_lib 
	cpu.cc
     )	
   add_library(concurrent_routines cpu.cc)
   
   #link cpu threading library
   target_link_libraries(
   concurrent_routines 
   # ${concurrent_routines_lib}
   ${CMAKE_THREAD_LIBS_INIT}
  )

  if(BuildShared)
	#install  libconcurrent_routines.a to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS concurrent_routines LIBRARY DESTINATION lib LIBRARY DESTINATION lib)
  else()
	# install  libconcurrent_routines.a to <cmake output dir>/${CMAKE_CURRENT_DIR}/
	install(TARGETS concurrent_routines LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  endif()

  #to do
  install(DIRECTORY ../include/concurrent_routines DESTINATION include)
  endif()


