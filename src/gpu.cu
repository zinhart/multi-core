#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/timer.hh"
#include <iostream>
namespace zinhart
{
  // KERNELS
  __global__ void parallel_saxpy_kernel(const float a, float * x, float * y, const std::uint32_t N)
  {
   	std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread_id > N)
	  return;
	y[thread_id] = a * x[thread_id] + y[thread_id];
  }

  
  // GPU WRAPPERS
  HOST int parallel_saxpy_gpu(
  		const float & a, float * x, float * y, const std::uint32_t N)
  {
	cudaError_t error_id;
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
  	dim3 block_launch;
	std::int32_t warp_size = properties.warpSize;
	std::int32_t threads_per_block = (N + warp_size -1) / warp_size * warp_size;
	if(threads_per_block > 4 * warp_size)
	  threads_per_block = 4 * warp_size;
	block_launch.x = (N + threads_per_block - 1) / threads_per_block;// number of blocks
	block_launch.y = 1;
	block_launch.z = 1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	parallel_saxpy_kernel<<<block_launch, threads_per_block >>>(a, x, y, N);
	cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
	error_id = cudaGetLastError();
  	if(error_id != cudaSuccess)
	{
	  std::cerr<<"saxpy failed to launch with error: "<<cudaGetErrorString(error_id)<<"\n";
	  return 1;
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"\ngpu time: "<<milliseconds;
	return 0;
  }
}
