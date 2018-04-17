# minimum cmake version
project(concurrent_routines)
cmake_minimum_required(VERSION 3.0)

# set build type to release by default
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, so project will be build without debug flags")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

# for unit tests using google test
option(BuildTest "BuildTest" ON)
# for cuda
option(BuildCuda "BuildCuda" ON)
# library type
option(BuildShared "BuildShared" OFF)
# sanitizer support
option(Sanitize "Sanitize" OFF)

# set lib type
if(BuildShared)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

# find cuda package and set compute achitecture
if(BuildCuda)
  find_package(CUDA REQUIRED)
  set(CUDA_ARCH -gencode arch = compute_61, code = sm_61)
endif()

# set include paths
include_directories( . include ${CUDA_INCLUDE_DIRS})

# set cxx standard
set(CMAKE_CXX_STANDARD 11)
# force cxx standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set cxx flags
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 ${CMAKE_CXX_FLAGS}")
# set cxx release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# set cxx debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -pedantic -fno-omit-frame-pointer -fno-inline-functions")

# set nvcc flags for debugging
if("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  if(BuildCuda)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -Xcompiler -rdynamic -Wno-deprecated-gpu-targets")
  endif()
endif()

# sanitize flags
if (Sanitize)
  set(CMAKE_CXX_FLAGS "-fsantize=address -fsanitize=undefined ${CMAKE_CXX_FLAGS}")
endif()

# add src directory
add_subdirectory(src)

#enable testing and add test directory
if(BuildTest)
  enable_testing()
  add_subdirectory(test)
endif()

message(STATUS "")
message(STATUS " BUILD SUMMARY")
message(STATUS " CMAKE_GENERATOR		  : ${CMAKE_GENERATOR}")
message(STATUS " CMAKE_GENERATOR		  : ${CMAKE_GENERATOR}")
message(STATUS " Compiler ID			  : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS " Build Type				  : ${CMAKE_BUILD_TYPE}")
message(STATUS " Library Type			  : ${LIB_TYPE}")
message(STATUS " Build Tests			  : ${BuildTest}")
message(STATUS " Sanitize Flags			  : ${Sanitize}")
if(BuildCuda)
  message(STATUS " Found CUDA			  : ${CUDA_FOUND}")
endif()
message(STATUS " CMAKE_CXX_FLAGS		  : ${Sanitize}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  message(STATUS " CMAKE_CXX_FLAGS		  : ${CMAKE_CXX_FLAGS_RELEASE}")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  message(STATUS " CMAKE_CXX_FLAGS_DEBUG  : ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
message(STATUS "")
