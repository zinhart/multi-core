#include "concurrent_routines/concurrent_routines_error.hh"
#include <iostream>
namespace zinhart
{
#if CUDA_ENABLED == 1
  HOST const char* cublas_get_error_string(cublasStatus_t status)
  {
	switch(status)
	{
	  case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
	  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
	  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_TATUS_MAPPING_ERROR";
	  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
	  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
	}
	return "unknown error";
  }

  HOST cudaError_t check_cuda_api(cudaError_t result)
  {
	if(result != cudaSuccess)
	  std::cerr<<"cuda runtime error: "<<cudaGetErrorString(result)<< " at "<<__FILE__<<" "<<__LINE__<<"\n";
	return result;
  }
  HOST cublasStatus_t check_cublas_api(cublasStatus_t result)
  {	
	if(result != CUBLAS_STATUS_SUCCESS)
	  std::cerr<<"cublas runtime error: "<<cublas_get_error_string(result)<< " at "<<__FILE__<<" "<<__LINE__<<"\n";
;
	return result;
  }
#endif
}
