#ifndef MACROS_HH
#define MACROS_HH

#if CUDA_ENABLED //this is defined in top level cmake lists file
  #define CUDA_CALLABLE_MEMBER __host__ __device__
  #define HOST __host__
  #define DEVICE __device__
  #define CONSTANT __constant__
  #define SHARED __shared__
  #include <cuda.h>
  #include <builtin_types.h>
  #include <cuda_runtime_api.h>
#else
  #define CUDA_ENABLED false
  #define CUDA_CALLABLE_MEMBER
  #define HOST
  #define DEVICE
  #define CONSTANT
  #define SHARED
#endif

#endif
