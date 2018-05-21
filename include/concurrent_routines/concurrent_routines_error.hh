#ifndef CONCURRENT_ROUTINES_ERROR_HH
#define CONCURRENT_ROUTINES_ERROR_HH
#include "macros.hh"
#if CUDA_ENABLED == true
#include <cublas_v2.h>
#endif
namespace zinhart
{
#if CUDA_ENABLED == true
  HOST const char * cublas_get_error_string(cublasStatus_t status);
  HOST cudaError_t check_cuda_api(cudaError_t result);
  HOST cublasStatus_t check_cublas_api(cublasStatus_t result);
#endif
}
#endif

