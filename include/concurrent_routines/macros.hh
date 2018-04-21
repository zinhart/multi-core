#ifndef MACROS_HH
#define MACROS_HH

//CUDACC is defined by nvcc
#ifdef __CUDACC__
#define CUDA_ENABLED true
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define HOST __host__
#define DEVICE __device__
#define CONSTANT __constant__
#define SHARED __shared_
#else
#define CUDA_ENABLED false
#define CUDA_CALLABLE_MEMBER
#define HOST
#define DEVICE
#define CONSTANT
#define SHARED
#endif

#include <cuda.h>
#include <builtin_types.h>

#endif
