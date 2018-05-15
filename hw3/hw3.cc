#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <random>
#include <limits>
#include <memory>
#include <iostream>
#include <iomanip>
#include "concurrent_routines/concurrent_routines.hh"
void execute_naive_matrix_mult(std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB);
void execute_shared_memory_matrix_mult(std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB);
void execute_naive_matrix_transpose(std::uint32_t N, std::uint32_t M);
void execute_shared_memory_matrix_transpose(std::uint32_t N, std::uint32_t M);
void execute_shared_memory_matrix_mult(std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB);
void execute_non_transposed_shared_memory_matrix_mult(std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB);
void execute_transposed_shared_memory_matrix_mult(std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB);


int main() 
{
  std::cout<<"NAIVE MEMORY MATRIX PRODUCT\n";

  execute_naive_matrix_mult(16, 16, 16, 1);
  execute_naive_matrix_mult(128, 128, 128, 1);
  execute_naive_matrix_mult(1024, 1024, 1024, 1); //error here all values <= 170 produces the error is an illegal memory access
  execute_naive_matrix_mult(2048, 2048, 2048, 1);
  execute_naive_matrix_mult(12048, 12048, 12048, 1);

  std::cout<<"SHARED MEMORY MATRIX PRODUCT\n";

  execute_shared_memory_matrix_mult(16, 16, 16, 1);
  execute_shared_memory_matrix_mult(128, 128, 128, 1);
  execute_shared_memory_matrix_mult(1024, 1024, 1024, 1);
  execute_shared_memory_matrix_mult(2048, 2048, 2048, 1);
  execute_shared_memory_matrix_mult(12048, 12048, 12048, 1);

  std::cout<<"NAIVE MEMORY MATRIX TRANPOSE\n";

  execute_naive_matrix_transpose(16, 16);
  execute_naive_matrix_transpose(128, 128);
  execute_naive_matrix_transpose(1024, 1024);
  execute_naive_matrix_transpose(2048, 2048);
  execute_naive_matrix_transpose(2048, 2048);
  execute_naive_matrix_transpose(12048, 12048);


  std::cout<<"SHARED MEMORY TRANSPOSE\n";

  execute_shared_memory_matrix_transpose(16, 16);
  execute_shared_memory_matrix_transpose(128, 128);
  execute_shared_memory_matrix_transpose(1024, 1024);
  execute_shared_memory_matrix_transpose(2048, 2048);
  execute_shared_memory_matrix_transpose(12048, 12048);
  
  std::cout<<"NON-TRANSPOSE (Regular) INDEXING MATRIX MULTIPLICATION\n";

  execute_non_transposed_shared_memory_matrix_mult(16, 16, 16, 1);
  execute_non_transposed_shared_memory_matrix_mult(128, 128, 128, 1);
  execute_non_transposed_shared_memory_matrix_mult(1024, 1024, 1024, 1);
  execute_non_transposed_shared_memory_matrix_mult(2048, 2048, 2048, 1);
  execute_non_transposed_shared_memory_matrix_mult(12048, 12048, 12048, 1);

  std::cout<<"TRANSPOSE INDEXING MATRIX MULTIPLICATION\n";

  execute_transposed_shared_memory_matrix_mult(16, 16, 16, 1);
  execute_transposed_shared_memory_matrix_mult(128, 128, 128, 1);
  execute_transposed_shared_memory_matrix_mult(1024, 1024, 1024, 1);
  execute_transposed_shared_memory_matrix_mult(2048, 2048, 2048, 1);
  execute_transposed_shared_memory_matrix_mult(12048, 12048, 12048, 1);

  return 0;
}

void execute_naive_matrix_mult(std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB)
{
  cudaError_t error_id;
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
  std::uint32_t A_n_elements = LDA * SDA;
  std::uint32_t B_n_elements = LDB * SDB;
  std::uint32_t C_n_elements = LDA * SDB;
  std::shared_ptr<float> A_host = std::shared_ptr<float>(new float [A_n_elements]);
  std::shared_ptr<float> B_host = std::shared_ptr<float>(new float [B_n_elements]);
  std::shared_ptr<float> C_host = std::shared_ptr<float>(new float [C_n_elements]);
  // device memory
  float * A_device, * B_device, * C_device;
  
  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &A_device, A_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"A_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &B_device, B_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"B_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &C_device, C_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"C_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";


  //randomly initialize A/B_host
  for(std::uint32_t i = 0; i < A_n_elements; ++i )
	A_host.get()[i] = real_dist(mt);
  for(std::uint32_t i = 0; i < B_n_elements; ++i )
	B_host.get()[i] = real_dist(mt);
  for(std::uint32_t i = 0; i < C_n_elements; ++i )
	C_host.get()[i] = 0.0f;


  //copy memory to device
  error_id = cudaMemcpy(A_device, A_host.get(), A_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"A_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  error_id = cudaMemcpy(B_device, B_host.get(), B_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"B_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  error_id = cudaMemcpy(C_device, C_host.get(), C_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"C_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  
  //call kernel
  zinhart::parallell_naive_matrix_product_gpu(A_device, B_device, C_device, LDA, SDA, LDB, SDB, LDA, SDB);
  //copy memory back to host and check for errors
  error_id = cudaMemcpy( A_host.get(), A_device, std::uint32_t(A_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"A_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( B_host.get(), B_device, std::uint32_t(B_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"B_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( C_host.get(), C_device, std::uint32_t(C_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"C_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 
 //free device memory
 cudaFree(A_device);
 if(error_id != cudaSuccess)
  std::cerr<<"A_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 cudaFree(B_device);
 if(error_id != cudaSuccess)
  std::cerr<<"B_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 cudaFree(C_device);
 if(error_id != cudaSuccess)
  std::cerr<<"C_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
}
void execute_shared_memory_matrix_mult(std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB)
{
  cudaError_t error_id;
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
  std::uint32_t A_n_elements = LDA * SDA;
  std::uint32_t B_n_elements = LDB * SDB;
  std::uint32_t C_n_elements = LDA * SDB;
  std::shared_ptr<float> A_host = std::shared_ptr<float>(new float [A_n_elements]);
  std::shared_ptr<float> B_host = std::shared_ptr<float>(new float [B_n_elements]);
  std::shared_ptr<float> C_host = std::shared_ptr<float>(new float [C_n_elements]);
  // device memory
  float * A_device, * B_device, * C_device;
  
  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &A_device, A_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"A_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &B_device, B_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"B_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &C_device, C_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"C_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";


  //randomly initialize A/B_host
  for(std::uint32_t i = 0; i < A_n_elements; ++i )
	A_host.get()[i] = real_dist(mt);
  for(std::uint32_t i = 0; i < B_n_elements; ++i )
	B_host.get()[i] = real_dist(mt);
  for(std::uint32_t i = 0; i < C_n_elements; ++i )
	C_host.get()[i] = 0.0f;


  //copy memory to device
  error_id = cudaMemcpy(A_device, A_host.get(), A_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"A_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  error_id = cudaMemcpy(B_device, B_host.get(), B_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"B_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  error_id = cudaMemcpy(C_device, C_host.get(), C_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"C_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  
  //call kernel
  zinhart::shared_matrix_product(A_device, B_device, C_device, LDA, SDA, LDB, SDB, LDA, SDB);

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( A_host.get(), A_device, std::uint32_t(A_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"A_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( B_host.get(), B_device, std::uint32_t(B_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"B_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( C_host.get(), C_device, std::uint32_t(C_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"C_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 
 //free device memory
 cudaFree(A_device);
 if(error_id != cudaSuccess)
  std::cerr<<"A_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 cudaFree(B_device);
 if(error_id != cudaSuccess)
  std::cerr<<"B_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 cudaFree(C_device);
 if(error_id != cudaSuccess)
  std::cerr<<"C_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
}
void execute_naive_matrix_transpose(std::uint32_t N, std::uint32_t M)
{
  cudaError_t error_id;
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
  std::uint32_t A_n_elements = N * M;
  std::shared_ptr<float> A_host = std::shared_ptr<float>(new float [A_n_elements]);
  // device memory
  float * A_out_device, * A_in_device;
  
  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &A_out_device, A_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"A_out_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &A_in_device, A_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"A_in_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //randomly initialize A
  for(std::uint32_t i = 0; i < A_n_elements; ++i )
	A_host.get()[i] = real_dist(mt);

  //copy memory to device
  error_id = cudaMemcpy(A_in_device, A_host.get(), A_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"A_in_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  
  //call kernel
  zinhart::parallell_naive_matrix_transpose_gpu(A_out_device, A_in_device, N, M);

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( A_host.get(), A_out_device, std::uint32_t(A_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"A_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 
   //free device memory
   cudaFree(A_out_device);
   if(error_id != cudaSuccess)
	std::cerr<<"A_out_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
   cudaFree(A_in_device);
   if(error_id != cudaSuccess)
	std::cerr<<"A_in_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
}

void execute_shared_memory_matrix_transpose(std::uint32_t N, std::uint32_t M)
{
  cudaError_t error_id;
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
  std::uint32_t A_n_elements = N * M;
  std::shared_ptr<float> A_host = std::shared_ptr<float>(new float [A_n_elements]);
  // device memory
  float * A_out_device, * A_in_device;
  
  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &A_out_device, A_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"A_out_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &A_in_device, A_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"A_in_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //randomly initialize A
  for(std::uint32_t i = 0; i < A_n_elements; ++i )
	A_host.get()[i] = real_dist(mt);

  //copy memory to device
  error_id = cudaMemcpy(A_in_device, A_host.get(), A_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"A_in_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  
  //call kernel
  zinhart::parallell_shared_matrix_transpose_gpu(A_out_device, A_in_device, N, M);

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( A_host.get(), A_out_device, std::uint32_t(A_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"A_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 
   //free device memory
   cudaFree(A_out_device);
   if(error_id != cudaSuccess)
	std::cerr<<"A_out_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
   cudaFree(A_in_device);
   if(error_id != cudaSuccess)
	std::cerr<<"A_in_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
}


void execute_non_transposed_shared_memory_matrix_mult(std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB)
{
  cudaError_t error_id;
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
  std::uint32_t A_n_elements = LDA * SDA;
  std::uint32_t B_n_elements = LDB * SDB;
  std::uint32_t C_n_elements = LDA * SDB;
  std::shared_ptr<float> A_host = std::shared_ptr<float>(new float [A_n_elements]);
  std::shared_ptr<float> B_host = std::shared_ptr<float>(new float [B_n_elements]);
  std::shared_ptr<float> C_host = std::shared_ptr<float>(new float [C_n_elements]);
  // device memory
  float * A_device, * B_device, * C_device;
  
  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &A_device, A_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"A_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &B_device, B_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"B_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &C_device, C_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"C_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";


  //randomly initialize A/B_host
  for(std::uint32_t i = 0; i < A_n_elements; ++i )
	A_host.get()[i] = real_dist(mt);
  for(std::uint32_t i = 0; i < B_n_elements; ++i )
	B_host.get()[i] = real_dist(mt);
  for(std::uint32_t i = 0; i < C_n_elements; ++i )
	C_host.get()[i] = 0.0f;


  //copy memory to device
  error_id = cudaMemcpy(A_device, A_host.get(), A_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"A_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  error_id = cudaMemcpy(B_device, B_host.get(), B_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"B_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  error_id = cudaMemcpy(C_device, C_host.get(), C_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"C_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  
  //call kernel
  zinhart::parallell_naive_matrix_product_gpu(A_device, B_device, C_device, LDA, SDA, LDB, SDB, LDA, SDB);
  //copy memory back to host and check for errors
  error_id = cudaMemcpy( A_host.get(), A_device, std::uint32_t(A_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"A_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( B_host.get(), B_device, std::uint32_t(B_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"B_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( C_host.get(), C_device, std::uint32_t(C_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"C_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 
 //free device memory
 cudaFree(A_device);
 if(error_id != cudaSuccess)
  std::cerr<<"A_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 cudaFree(B_device);
 if(error_id != cudaSuccess)
  std::cerr<<"B_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 cudaFree(C_device);
 if(error_id != cudaSuccess)
  std::cerr<<"C_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
}

void execute_transposed_shared_memory_matrix_mult(std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB)
{
  cudaError_t error_id;
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
  std::uint32_t A_n_elements = LDA * SDA;
  std::uint32_t B_n_elements = LDB * SDB;
  std::uint32_t C_n_elements = LDA * SDB;
  std::shared_ptr<float> A_host = std::shared_ptr<float>(new float [A_n_elements]);
  std::shared_ptr<float> B_host = std::shared_ptr<float>(new float [B_n_elements]);
  std::shared_ptr<float> C_host = std::shared_ptr<float>(new float [C_n_elements]);
  // device memory
  float * A_device, * B_device, * C_device;
  
  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &A_device, A_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"A_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &B_device, B_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"B_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //allocate device memory and check for errors
  error_id = cudaMalloc( (void **) &C_device, C_n_elements * sizeof(float) );
  if(error_id != cudaSuccess)
	std::cerr<<"C_device memory alloc failed with error: "<<cudaGetErrorString(error_id)<<"\n";


  //randomly initialize A/B_host
  for(std::uint32_t i = 0; i < A_n_elements; ++i )
	A_host.get()[i] = real_dist(mt);
  for(std::uint32_t i = 0; i < B_n_elements; ++i )
	B_host.get()[i] = real_dist(mt);
  for(std::uint32_t i = 0; i < C_n_elements; ++i )
	C_host.get()[i] = 0.0f;


  //copy memory to device
  error_id = cudaMemcpy(A_device, A_host.get(), A_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"A_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  error_id = cudaMemcpy(B_device, B_host.get(), B_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"B_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  error_id = cudaMemcpy(C_device, C_host.get(), C_n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"C_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  
  //call kernel
  zinhart::shared_matrix_product(A_device, B_device, C_device, LDA, SDA, LDB, SDB, LDA, SDB);

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( A_host.get(), A_device, std::uint32_t(A_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"A_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( B_host.get(), B_device, std::uint32_t(B_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"B_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //copy memory back to host and check for errors
  error_id = cudaMemcpy( C_host.get(), C_device, std::uint32_t(C_n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  if(error_id != cudaSuccess)
	std::cerr<<"C_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 
 //free device memory
 cudaFree(A_device);
 if(error_id != cudaSuccess)
  std::cerr<<"A_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 cudaFree(B_device);
 if(error_id != cudaSuccess)
  std::cerr<<"B_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
 cudaFree(C_device);
 if(error_id != cudaSuccess)
  std::cerr<<"C_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
}
