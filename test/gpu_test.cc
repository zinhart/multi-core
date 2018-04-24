#include "concurrent_routines/concurrent_routines.hh"
#include "gtest/gtest.h"
#include <random>
#include <limits>
#include <iostream>
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

  //validate each value
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_host.get()[i], y_host_copy.get()[i]);
  }
  cudaFree(x_device);
  if(error_id != cudaSuccess)
	std::cerr<<"x_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  cudaFree(y_device);
  if(error_id != cudaSuccess)
	std::cerr<<"y_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  std::cout<<"Hello From GPU Tests\n";
}
