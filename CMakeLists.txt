# minimum cmake version
cmake_minimum_required(VERSION 3.1)
# library name
project(concurrent_routines)
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

#threading options 
option(PTHREADS "PTHREADS" ON)
option(WIN32_THREADS "WIN32_THREADS" OFF)

option (BuildHw2 "BuildHw2" OFF)
option (BuildHw3 "BuildHw3" OFF)

# set lib type
if(BuildShared)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

#set module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(PTHREADS)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
elseif(WIN32_THREADS)
  set(THREADS_PREFER_WIN32_FLAG ON)
elseif(PTHREADS AND WIN32_THREADS)
  message(FATAL_ERROR "Choose 1 threading library")
endif()

# threading library
find_package(Threads REQUIRED)
# lapacke for matrix multiplications
find_package(LAPACK REQUIRED)


# find cuda package and set compute achitecture
if(BuildCuda)
  find_package(CUDA REQUIRED)
  set(CUDA_ARCH -gencode arch = compute_61, code = sm_61)
  # set macro to conditional include headers define functions etc w.r.t gpu code
  add_definitions(-DCUDA_ENABLED=true)
else()
	add_definitions(-DCUDA_ENABLED=false)
endif()

# set include paths
include_directories( . include ${CUDA_INCLUDE_DIRS} "${PROJECT_BINARY_DIR}")

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
  #only if Building gpu support
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

if(BuildHw2)
  add_subdirectory(hw2)
endif()

if(BuildHw3)
  add_subdirectory(hw3)
endif()

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
message(STATUS " Build Type			  : ${CMAKE_BUILD_TYPE}")
message(STATUS " Library Type		  : ${LIB_TYPE}")
message(STATUS " Build Tests			  : ${BuildTest}")
message(STATUS " Sanitize Flags		  : ${Sanitize}")
if(BuildCuda)
  message(STATUS " Found CUDA			  : ${CUDA_FOUND}")
endif()
message(STATUS " CMAKE_CXX_FLAGS		  : ${CMAKE_CXX_FLAGS}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
	message(STATUS " CMAKE_CXX_RELEASE_FLAGS		  : ${CMAKE_CXX_FLAGS_RELEASE}")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  message(STATUS " CMAKE_CXX_FLAGS_DEBUG         : ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
message(STATUS "")
