#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_error.hh"
#include "gtest/gtest.h"
#include <random>
#include <limits>
#include <iostream>
#if CUDA_ENABLED == true
#include <cublas_v2.h>
#endif
TEST(gpu_test, parrallel_saxpy_gpu)
{
#if CUDA_ENABLED
  std::cout<<"CUDA ENABLEs\n";
#else
  std::cout<<"Cuda Disabled\n";
#endif
  cudaError_t error_id;
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_host = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_host = std::shared_ptr<float>(new float [n_elements]);
  //the device memory will be copied to here
  std::shared_ptr<float> y_host_copy = std::shared_ptr<float>(new float [n_elements]);

  float * x_device, * y_device; 
  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &x_device, n_elements * sizeof(float) );
  //check for errors
  if(error_id != cudaSuccess)
	std::cerr<<"x_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &y_device, n_elements * sizeof(float) );
  //check for errors
  if(error_id != cudaSuccess)
	std::cerr<<"y_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  auto serial_saxpy = [](const float & a, float & x, float & y)
						{
						  return a * x + y;
						};

  const float a = real_dist(mt);
  std::uint32_t i = 0;
  //randomly initialize x_host/(_copy)
  for(i = 0; i < n_elements; ++i )
  {
	x_host.get()[i] = real_dist(mt);
	y_host.get()[i] = real_dist(mt);
	y_host_copy.get()[i] = y_host.get()[i];
	y_host.get()[i] = serial_saxpy(a, x_host.get()[i], y_host.get()[i]);
  }

  //copy memory to device
  error_id = cudaMemcpy(y_device, y_host_copy.get(), n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"y_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaMemcpy(x_device, x_host.get(), n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"x_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //call kernel
  zinhart::parallel_saxpy_gpu(a, x_device, y_device, n_elements);
 
  
  //copy memory back to host
  error_id = cudaMemcpy( y_host_copy.get(), y_device, std::uint32_t(n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  //check for errors
  if(error_id != cudaSuccess)
	std::cerr<<"y_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n"; 

  std::cout<<"N elements "<<n_elements<<"\n";
  //validate each value
  float epsilon = 0.0005;
  for(i = 0; i < n_elements; ++i)
  {
	//std::cout<<y_host.get()[i]<<" "<<y_host_copy.get()[i]<<"\n";
	ASSERT_NEAR(y_host.get()[i], y_host_copy.get()[i], epsilon);
  }
  cudaFree(x_device);
  if(error_id != cudaSuccess)
	std::cerr<<"x_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  cudaFree(y_device);
  if(error_id != cudaSuccess)
	std::cerr<<"y_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  std::cout<<"Hello From GPU Tests\n";
}
TEST(gpu_test, gemm_wrapper)
{
  cudaError_t error_id;
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint8_t> uint_dist(1, std::numeric_limits<std::uint8_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
  std::int32_t A_row = uint_dist(mt);
  std::int32_t A_col = uint_dist(mt);
  std::int32_t A_total_elements = A_row  * A_col;
  std::int32_t B_row = A_col;
  std::int32_t B_col = uint_dist(mt);
  std::int32_t B_total_elements = B_row * B_col;
  std::int32_t C_row = A_row;
  std::int32_t C_col = B_row;
  std::int32_t C_total_elements = C_row * C_col;
  std::cout<<"total pinned memory: "<<std::uint32_t(A_total_elements * B_total_elements * C_total_elements)<<" bytes.\n";
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
							  double sum = 0.0;
							  for(std::uint32_t k = 0; k < A_cols; ++k)
							  {
								float a = A[i * A_cols + k];
								float b = B[k * B_cols + j];
								C[i * B_cols + j] += a * b;
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

  for(std::uint32_t i = 0; i < A_total_elements; ++i)
  {
	A_host[i] = real_dist(mt);
	A_host_copy[i] = 0;
  }
  for(std::uint32_t i = 0; i < B_total_elements; ++i)
  {
	B_host[i] = real_dist(mt);
	B_host_copy[i] = 0;
  }
  for(std::uint32_t i = 0; i < C_total_elements; ++i)
  {
	C_host[i] = 0.0f;
	C_host_copy[i] = 0;
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
  /*zinhart::check_cublas_api(cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N,
			  m, n, k,
			  &alpha,
			  B_device, lda,
			  A_device, ldb,
			  &beta,
			  C_device, ldc
			 ));*/

  
  zinhart::check_cuda_api(cudaMemcpy(A_host_copy, A_device, A_total_elements * sizeof(double), cudaMemcpyDeviceToHost));
  zinhart::check_cuda_api(cudaMemcpy(B_host_copy, B_device, B_total_elements * sizeof(double), cudaMemcpyDeviceToHost));
  zinhart::check_cuda_api(cudaMemcpy(C_host_copy, C_device, C_total_elements * sizeof(double), cudaMemcpyDeviceToHost));
  zinhart::check_cublas_api(cublasDestroy(context));

  //matrix_product(A_host, B_host, C_host, A_row, A_col, B_col);
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

