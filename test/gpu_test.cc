#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_error.hh"
#include "gtest/gtest.h"
#include <random>
#include <limits>
#include <iostream>
#if CUDA_ENABLED == true
#include <cublas_v2.h>
#endif
TEST(gpu_test, gemm_wrapper)
{
  cudaError_t error_id;
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint8_t> uint_dist(1, std::numeric_limits<std::uint8_t>::max() );
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
  std::int32_t A_row = uint_dist(mt);
  std::int32_t A_col = uint_dist(mt);
  std::int32_t A_total_elements = A_row  * A_col;
  std::int32_t B_row = A_col;
  std::int32_t B_col = uint_dist(mt);
  std::int32_t B_total_elements = B_row * B_col;
  std::int32_t C_row = A_row;
  std::int32_t C_col = B_col;
  std::int32_t C_total_elements = C_row * C_col;
  std::cout<<"total pinned memory: "<<std::uint32_t(A_total_elements * B_total_elements * C_total_elements)<<" bytes.\n";
  std::cout<<" ";
  std::int32_t m, n, k, lda, ldb, ldc;
  double * A_host;
  double * B_host;
  double * C_host;
  double * A_host_copy;
  double * B_host_copy;
  double * C_host_copy;
  double * A_device, * B_device, * C_device;
  auto matrix_product = [](double * A, double * B, double * C, std::uint32_t A_rows, std::uint32_t A_cols, std::uint32_t B_cols)
						{
						  for(std::uint32_t i = 0; i < A_rows; ++i)
						  {
							for(std::uint32_t j = 0; j < B_cols; ++j)
							{
							  for(std::uint32_t k = 0; k < A_cols; ++k)
							  {
								float a = A[zinhart::idx2r(i, k, A_cols)];
								float b = B[zinhart::idx2r(k, j, B_cols)];
								C[zinhart::idx2r(i, j, B_cols)] += a * b;
							  } 
							}
						  }
						};
  auto print_mat = [](double * mat, std::uint32_t mat_rows, std::uint32_t mat_cols, std::string s)
			   {
				 std::cout<<s<<"\n";
	   			 for(std::uint32_t i = 0; i < mat_rows; ++i)  
				 {
				   for(std::uint32_t j = 0; j < mat_cols; ++j)
				   {
					 std::cout<<mat[i * mat_cols + j]<<" ";
				   }
				   std::cout<<"\n";
				 }
				 
			   };

  zinhart::check_cuda_api(cudaHostAlloc((void**)&A_host, A_total_elements * sizeof(double), cudaHostAllocDefault));
  zinhart::check_cuda_api(cudaHostAlloc((void**)&B_host, B_total_elements * sizeof(double), cudaHostAllocDefault));
  zinhart::check_cuda_api(cudaHostAlloc((void**)&C_host, C_total_elements * sizeof(double), cudaHostAllocDefault));

  zinhart::check_cuda_api(cudaHostAlloc((void**)&A_host_copy, A_total_elements * sizeof(double), cudaHostAllocDefault));
  zinhart::check_cuda_api(cudaHostAlloc((void**)&B_host_copy, B_total_elements * sizeof(double), cudaHostAllocDefault));
  zinhart::check_cuda_api(cudaHostAlloc((void**)&C_host_copy, C_total_elements * sizeof(double), cudaHostAllocDefault));

  for (std::uint32_t i =0; i < A_row; ++i)
  {
	for(std::uint32_t j = 0; j < A_col; ++j)
	{
	  A_host[zinhart::idx2r(i,j, A_col)] = real_dist(mt);
	}
  }

  for (std::uint32_t i =0; i < B_row; ++i)
  {
	for(std::uint32_t j = 0; j < B_col; ++j)
	{
	  B_host[zinhart::idx2r(i,j, B_col)] = real_dist(mt);
	}
  }

  for (std::uint32_t i =0; i < C_row; ++i)
  {
	for(std::uint32_t j = 0; j < C_col; ++j)
	{
	  C_host[zinhart::idx2r(i,j, C_col)] = 0.0f;
	}
  }
/*  print_mat(A_host, A_row, A_col,"A_host");
  print_mat(B_host, B_row, B_col,"B_host");
  print_mat(C_host, C_row, C_col,"C_host");*/

  for(std::uint32_t i = 0; i < A_total_elements; ++i)
  {
	A_host_copy[i] = 0;
  }
  for(std::uint32_t i = 0; i < B_total_elements; ++i)
  {
	B_host_copy[i] = 0;
  }
  for(std::uint32_t i = 0; i < C_total_elements; ++i)
  {
	C_host_copy[i] = 0.0f;
  }

  zinhart::check_cuda_api(cudaMalloc((void**)&A_device,  A_total_elements * sizeof(double)));
  zinhart::check_cuda_api(cudaMalloc((void**)&B_device,  B_total_elements * sizeof(double)));
  zinhart::check_cuda_api(cudaMalloc((void**)&C_device,  C_total_elements * sizeof(double)));


  zinhart::check_cuda_api(cudaMemcpy(A_device, A_host, A_total_elements * sizeof(double), cudaMemcpyHostToDevice));
  zinhart::check_cuda_api(cudaMemcpy(B_device, B_host, B_total_elements * sizeof(double), cudaMemcpyHostToDevice));
  zinhart::check_cuda_api(cudaMemcpy(C_device, C_host, C_total_elements * sizeof(double), cudaMemcpyHostToDevice));

  cublasStatus_t cublas_error_id;
  cublasHandle_t context;
  zinhart::check_cublas_api(cublasCreate(&context));
  double alpha = 1;
  double beta = 1; 
  zinhart::gemm_wrapper(m,n,k,lda,ldb,ldc, A_row, A_col, B_row, B_col); 
  //sgemm here
  zinhart::check_cublas_api(cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N,
			  m, n, k,
			  &alpha,
			  B_device, lda,
			  A_device, ldb,
			  &beta,
			  C_device, ldc
			 ));/**/

  
  zinhart::check_cuda_api(cudaMemcpy(A_host_copy, A_device, A_total_elements * sizeof(double), cudaMemcpyDeviceToHost));
  zinhart::check_cuda_api(cudaMemcpy(B_host_copy, B_device, B_total_elements * sizeof(double), cudaMemcpyDeviceToHost));
  zinhart::check_cuda_api(cudaMemcpy(C_host_copy, C_device, C_total_elements * sizeof(double), cudaMemcpyDeviceToHost));
  zinhart::check_cublas_api(cublasDestroy(context));

  matrix_product(A_host, B_host, C_host, A_row, A_col, B_col);
/*  print_mat(A_host, A_row, A_col,"A_host");
  print_mat(A_host_copy, A_row, A_col, "A_host_copy");
  print_mat(B_host, B_row, B_col,"B_host");
  print_mat(B_host_copy, B_row, B_col, "B_host_copy");
  print_mat(C_host, C_row, C_col,"C_host");
  print_mat(C_host_copy, C_row, C_col, "C_host_copy");*/
  double epsilon = .0005;
  for(std::uint32_t i = 0; i < A_total_elements; ++i)
  {
	ASSERT_NEAR(A_host[i], A_host_copy[i], epsilon);
  }
  for(std::uint32_t i = 0; i < B_total_elements; ++i)
  {
	ASSERT_NEAR(B_host[i], B_host_copy[i], epsilon);
  }
  for(std::uint32_t i = 0; i < C_total_elements; ++i)
  {
	ASSERT_NEAR(C_host[i],C_host_copy[i], epsilon);
  }

  zinhart::check_cuda_api(cudaFreeHost(A_host));
  zinhart::check_cuda_api(cudaFreeHost(B_host));
  zinhart::check_cuda_api(cudaFreeHost(C_host));
  zinhart::check_cuda_api(cudaFreeHost(A_host_copy));
  zinhart::check_cuda_api(cudaFreeHost(B_host_copy));
  zinhart::check_cuda_api(cudaFreeHost(C_host_copy));
  zinhart::check_cuda_api(cudaFree(A_device));
  zinhart::check_cuda_api(cudaFree(B_device));
  zinhart::check_cuda_api(cudaFree(C_device)); 
}

