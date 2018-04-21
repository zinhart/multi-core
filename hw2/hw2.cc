// Note: Most of the code comes from the MacResearch OpenCL podcast

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "concurrent_routines/concurrent_routines.hh"
#include "mandelbrot/hw2.hh"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <random>
#include <limits>
#include <memory>
#include <iostream>
#include <iomanip>
/*void compare_saxpy(std::uint32_t n_elements);
void compare_mandelbrot(std::uint32_t width, std::uint32_t height, std::uint32_t iters);

class Timer {
  typedef std::chrono::high_resolution_clock high_resolution_clock;
  typedef std::chrono::milliseconds milliseconds;
  public:
  explicit Timer(bool run = false)
  {
	if (run)
	  Reset();
  }
  void Reset()
  {
	_start = high_resolution_clock::now();
  }
  milliseconds Elapsed() const
  {
	return std::chrono::duration_cast<milliseconds>(high_resolution_clock::now() - _start);
  }
  template <typename T, typename Traits>
	friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, const Timer& timer)
	{
	  return out << timer.Elapsed().count();	
	}
  private:
  high_resolution_clock::time_point _start;
};*/


int main() 
{
  mandelbrot(7680, 7680, 8192);

  /*
  compare_saxpy(16);
  compare_saxpy(128);
  compare_saxpy(1024);
  compare_saxpy(2048);
  compare_saxpy(8192);
  compare_saxpy(65536);

  compare_mandelbrot(1024, 1024, 512);
  compare_mandelbrot(2048, 2048, 512);
  compare_mandelbrot(4096, 4096, 512);
  compare_mandelbrot(8192, 8192, 512);*/
  return 0;
}
/*
void compare_saxpy(std::uint32_t n_elements)
{
  auto serial_saxpy = [](const float & a, float * x, float * y,std::uint32_t n_elem)
				{
				  	 for(std::uint32_t i  = 0; i < n_elem; ++i)
					 {
					   y[i] = a * x[i] + y[i];
					 } 
				};
  cudaError_t error_id;
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);

  std::shared_ptr<float> x_host = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_host = std::shared_ptr<float>(new float [n_elements]);
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

  const float a = real_dist(mt);
  std::uint32_t i = 0;
  //randomly initialize x_host/(_copy)
  for(i = 0; i < n_elements; ++i )
  {
	x_host.get()[i] = real_dist(mt);
	y_host.get()[i] = real_dist(mt);
	y_host_copy.get()[i] = y_host.get()[i];
  }

  //copy memory to device
  error_id = cudaMemcpy(y_device, y_host_copy.get(), n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"y_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  error_id = cudaMemcpy(x_device, x_host.get(), n_elements * sizeof(float), cudaMemcpyHostToDevice);
  if(error_id != cudaSuccess)
	std::cerr<<"x_device (HostToDevice) memcpy failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  //timers here
  Timer timer_gpu(true);
  //call kernel
  zinhart::parallel_saxpy_gpu(a, x_device, y_device, n_elements);
  auto elapsed_gpu = timer_gpu.Elapsed();
  Timer timer_cpu(true);
  serial_saxpy(a,x_host.get(),y_host.get(),n_elements);
  auto elapsed_cpu = timer_cpu.Elapsed(); 
  std::cout<<std::fixed<<"\nN:" <<n_elements<<"\ngpu time"<<elapsed_gpu.count()<<"ms "<<"cpu time"<<elapsed_cpu.count()<<"ms";

 
  //copy memory back to host
  error_id = cudaMemcpy( y_host_copy.get(), y_device, std::uint32_t(n_elements) * sizeof(float), cudaMemcpyDeviceToHost);
  //check for errors
  if(error_id != cudaSuccess)
	std::cerr<<"y_host (DeviceToHost) failed with error: "<<cudaGetErrorString(error_id)<<"\n";

  cudaFree(x_device);
  if(error_id != cudaSuccess)
	std::cerr<<"x_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  cudaFree(y_device);
  if(error_id != cudaSuccess)
	std::cerr<<"y_device deallocation failed with error: "<<cudaGetErrorString(error_id)<<"\n";
  
}
void compare_mandelbrot(std::uint32_t width, std::uint32_t height, std::uint32_t iters)
{
  Timer timer_gpu(true);
  mandelbrot(width, height, iters);
  auto elapsed_gpu = timer_gpu.Elapsed();
  std::cout<<"\n"<<std::fixed<<"width:" <<width<<" height: "<<height<<" iterations: "<<iters<<" gpu time: "<<elapsed_gpu.count()<<" ms";
}
*/
