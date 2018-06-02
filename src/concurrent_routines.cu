#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_error.hh"
#include "concurrent_routines/timer.hh"
#include <iostream>
namespace zinhart
{
  //EXPLICIT INSTANTIATIONS (to make templates , such as the wrapper functions, in defined in the .cu available in main.cc)
  
  template HOST std::int32_t call_axps(const float & a, float * x, const float & s, const std::uint32_t & N, const std::uint32_t & device_id);
  template HOST std::int32_t call_axps(const double & a, double * x, const double & s, const std::uint32_t & N, const std::uint32_t & device_id);

  template HOST std::int32_t call_axps_async(const float & a, float * x, const float & s, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);
  template HOST std::int32_t call_axps_async(const double & a, double * x, const double & s, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);
  // KERNELS
  
  template <class Precision_Type>
	__global__ void axps(Precision_Type a, Precision_Type * x, Precision_Type s, std::uint32_t N)
	{
	  for(std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;  thread_id < N; thread_id += blockDim.x * gridDim.x)
		x[thread_id] = a * x[thread_id] + s;
	}
  template <class Precision_Type>
	__global__ void axps(const Precision_Type a, Precision_Type * x, const Precision_Type s, std::uint32_t N, const std::uint32_t shared_memory_length)
	{
	  extern __shared__  std::uint8_t x_shared[];
	  Precision_Type * x_tile = reinterpret_cast<Precision_Type*>(x_shared);
	  // grid-stride loop
	  for(std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;  thread_id < N; thread_id += blockDim.x * gridDim.x)
	  {
		x_tile[thread_id] = x[thread_id];
		x_tile[thread_id] = a * x_tile[thread_id ] + s;
  		x[thread_id] = x_tile[thread_id];
	  }
	}

  // GPU WRAPPERS
  
  template <class Precision_Type>
	HOST std::int32_t call_axps(const Precision_Type & a, Precision_Type * x, const Precision_Type & s, const std::uint32_t & N, const std::uint32_t & device_id)
	{
	  dim3 num_blocks;
	  dim3 threads_per_block;
	  grid_space::get_launch_params(num_blocks, threads_per_block, N, device_id);
	  std::cout<<"num_blocks.x: "<<num_blocks.x<<" num_blocks.y: "<<num_blocks.y<<" num_blocks.z: "<<num_blocks.z<<" threads_per_block: "<<threads_per_block.x<<" N:" <<N<<"\n";
	  // call kernel
	  axps<<<num_blocks,threads_per_block>>>(a,x,s,N);
	  return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()), __FILE__,__LINE__);
	}
  template <class Precision_Type>
	HOST std::int32_t call_axps_async(const Precision_Type & a, Precision_Type * x, const Precision_Type & s, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id)
	{
	  dim3 num_blocks;
	  dim3 threads_per_block;
	  std::uint32_t shared_memory_bytes{0};
	  grid_space::get_launch_params(num_blocks, threads_per_block, N, shared_memory_bytes, device_id, Precision_Type{});
	  // here will have either the max amount of shared memory or less;
	  const std::uint32_t shared_memory_length = shared_memory_bytes / sizeof(Precision_Type);
	 /* std::cout<<"N:" <<N<<"\n";
	  std::cout<<"shared_memory: "<<shared_memory_length<<"\n";
	  std::cout<<"num_blocks.x: "<<num_blocks.x<<" num_blocks.y: "<<num_blocks.y<<" num_blocks.z: "<<num_blocks.z<<"\n";
	  std::cout<<"threads_per_block.x: "<<threads_per_block.x<<" threads_per_block.y: "<<threads_per_block.y<<" threads_per_block.z: "<< threads_per_block.z<<"\n";	 */ 
	  axps<<<num_blocks, threads_per_block, shared_memory_bytes, stream>>>(a, x, s, N, shared_memory_length);
	  return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()), __FILE__,__LINE__);
	}

  //to do 
  template <class Precision_Type>
	void reduce(std::uint32_t size, std::uint32_t threads, std::uint32_t blocks, Precision_Type * out, Precision_Type * in)
	  {
		dim3 dimBlock(threads, 1, 1);
		dim3 dimGrid(blocks, 1, 1);

	    // when there is only one warp per block, we need to allocate two warps
		// worth of shared memory so that we don't index shared memory out of bounds
		//std::uint32_t shared_memory = (threads <= 32) ? 2 * threads * sizeof(Precision_Type) : threads * sizeof(Precision_Type);
	  }

	namespace cuda_device_properties
	{
	  HOST auto get_properties(std::uint32_t device_id) -> cudaDeviceProp
	  {
		static cudaDeviceProp properties;
		check_cuda_api(cudaGetDeviceProperties(&properties, device_id),__FILE__, __LINE__);
		return properties;

	  }	
	  HOST void get_warp_size(std::uint32_t & warp_size, const std::uint32_t & device_id)
	  {
		warp_size = get_properties(device_id).warpSize;
	  }
	  HOST void get_max_shared_memory(std::uint32_t & max_shared_memory_per_block, const std::uint32_t & device_id)
	  {
		max_shared_memory_per_block = get_properties(device_id).sharedMemPerBlock;
	  }
	  HOST void get_max_threads_per_block(std::uint32_t & max_threads_per_block, const std::uint32_t & device_id)
	  {
		max_threads_per_block = get_properties(device_id).maxThreadsPerBlock;
	  }
	  HOST void get_max_threads_dim(std::int32_t (& max_threads_dim)[3], const std::uint32_t & device_id)
	  {
		max_threads_dim[0] = get_properties(device_id).maxThreadsDim[0];
		max_threads_dim[1] = get_properties(device_id).maxThreadsDim[1];
		max_threads_dim[2] = get_properties(device_id).maxThreadsDim[2];
	  }
	  HOST void get_max_grid_size(std::int32_t (& max_grid_size)[3], const std::uint32_t & device_id)
	  {
		max_grid_size[0] = get_properties(device_id).maxGridSize[0];
		max_grid_size[1] = get_properties(device_id).maxGridSize[1];
		max_grid_size[2] = get_properties(device_id).maxGridSize[2];
	  }
	}
	namespace grid_space
	{
	  HOST bool get_launch_params(dim3 & num_blocks, dim3 & threads_per_block, std::uint32_t N, const std::uint32_t & device_id)
	  {
		std::uint64_t max_outputs_1d_kernel{0};
		std::uint64_t max_outputs_2d_kernel{0};
		std::uint64_t max_outputs_3d_kernel{0};
		cuda_device_properties::max_threads<1>::get_max(max_outputs_1d_kernel, device_id); 
		cuda_device_properties::max_threads<2>::get_max(max_outputs_2d_kernel, device_id); 
		cuda_device_properties::max_threads<3>::get_max(max_outputs_3d_kernel, device_id); 
		if(N <= max_outputs_1d_kernel)
		{
		  grid<1>::get_launch_params(num_blocks, threads_per_block, N, device_id);
		  return false;
		}
		else if (N <= max_outputs_2d_kernel && N > max_outputs_1d_kernel)
		{
		  return false;
		}
		else if(N <= max_outputs_3d_kernel && N > max_outputs_2d_kernel)
		{
		  return false;
		}
		else
		{
		return true;
		}
	  }
	  
	}
}
