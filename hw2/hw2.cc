#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/timer.hh"
#include "mandelbrot/hw2.hh"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <random>
#include <limits>
#include <memory>
#include <iostream>
#include <iomanip>
void compare_saxpy(std::uint32_t n_elements);
void compare_mandelbrot(std::uint32_t width, std::uint32_t height, std::uint32_t iters);
void serial_render ( char *out , const  int width , const  int  height  , const  int  max_iter);

int main() 
{
  std::cout<<"All time units should be interpreted as milliseconds";
  std::cout<<"\nSaxpy timed";
  compare_saxpy(16);
  compare_saxpy(128);
  compare_saxpy(1024);
  compare_saxpy(2048);
  compare_saxpy(8192);
  compare_saxpy(65536);
  compare_saxpy(1000000);

  std::cout<<"\nMandelbrot timed";
  compare_mandelbrot(1024, 1024, 512);
  compare_mandelbrot(2048, 2048, 512);
  compare_mandelbrot(4096, 4096, 512);
  compare_mandelbrot(8192, 8192, 512);
  std::cout<<"\n";
  return 0;
}

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
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  serial_saxpy(a,x_host.get(),y_host.get(),n_elements);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout<<"\nN: "<<n_elements;
  std::cout<<"\ncpu time: "<<milliseconds;

  //call kernel
  zinhart::parallel_saxpy_gpu(a, x_device, y_device, n_elements);

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
  std::cout<<"\nwidth: "<<width<< " height: " <<height<<" iterations: "<<iters;
  mandelbrot(width, height, iters);
  size_t buffer_size = sizeof(char) * width * height * 3;
  char *host_image = (char *) malloc(buffer_size);
/*
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  serial_render(host_image, width, height, buffer_size);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout<<"\ncpu time: "<<milliseconds;
  write_bmp("output_serial.bmp", width, height, host_image);
  free(host_image);
  */
}

void serial_render ( char *out , const  int width , const  int  height  , const  int  max_iter)
{
  float  x_origin , y_origin , xtemp , x, y;
  int  iteration , index;
  for(int i = 0; i < width; i++)
  {
   	for(int j = 0; j < height; j++)
	{
	  index = 3* width*j + i*3;
	  iteration = 0;
	  x = 0.0f;
	  y = 0.0f;
	  x_origin = (( float) i/width)*3.25f  -2.0f;
	  y_origin = (( float) j/width)*2.5f - 1.25f;
	  while(x*x + y*y  <= 4 &&  iteration  < max_iter)
	  {
	  xtemp = x*x - y*y + x_origin;
	  y = 2*x*y + y_origin;
	  x = xtemp;
	  iteration ++;
	  }
	  if(iteration == max_iter)
	  {
		out[index] = 0;
		out[index + 1] = 0;
		out[index + 2] = 0;
	  }
	  else
	  {
		out[index] = iteration;
		out[index + 1] = iteration;
		out[index + 2] = iteration;
	  }
	}
  }
}

