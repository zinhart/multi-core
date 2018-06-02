#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/timer.hh"
#include <iostream>
namespace zinhart
{
  //EXPLICIT INSTANTIATIONS (to make templates , such as the wrapper functions, in defined in the .cu available in main.cc)
  
  template HOST std::int32_t call_axps(float a, float * x, float s, std::uint32_t N, const std::uint32_t & device_id);
  template HOST std::int32_t call_axps(double a, double * x, double s, std::uint32_t N, const std::uint32_t & device_id);

  template HOST std::int32_t call_axps_async(float a, float * x, float s, std::uint32_t N, cudaStream_t & stream, const std::uint32_t & device_id);
  template HOST std::int32_t call_axps_async(double a, double * x, double s, std::uint32_t N, cudaStream_t & stream, const std::uint32_t & device_id);
  // KERNELS
  
  template <class Precision_Type>
	__global__ void axps(Precision_Type a, Precision_Type * x, Precision_Type s, std::uint32_t N)
	{
	  const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	  if(thread_id > N )
		return;
	  x[thread_id] = a * x[thread_id] + s;
	}
  template <class Precision_Type>
	__global__ void axps(Precision_Type a, Precision_Type * x, Precision_Type s, std::uint32_t N, const std::uint32_t tile_dim, const std::uint32_t block_row)
	{
	  extern __shared__  std::uint8_t x_shared[];
	  Precision_Type * x_tile = reinterpret_cast<Precision_Type*>(x_shared);
	  const std::uint32_t thread_id = blockIdx.x * tile_dim + threadIdx.x;
	  const std::uint32_t width = gridDim.x * tile_dim;
	  if(thread_id > N )
		return;
	  /*for(std::uint32_t i = 0; i < tile_dim; i+= block_row)
	  {
		x_tile[(threadIdx.y + i) * tile_dim + threadIdx.x] = x[ (thread_id_y + i) * width + thread_id_x];
		x_tile[(threadIdx.y + i) * tile_dim + threadIdx.x] = a * x_tile[(threadIdx.y + i) * tile_dim + threadIdx.x] + s;
	  }
	  for(std::uint32_t i = 0; i < tile_dim; i+= block_row)
	  {
		x[(thread_id_y+i) * width + thread_id_x] = x_tile[(threadIdx.y + i) * tile_dim + threadIdx.x];
	  }*/
	}

  // GPU WRAPPERS
  
  template <class Precision_Type>
	HOST std::int32_t call_axps(Precision_Type a, Precision_Type * x, Precision_Type s, std::uint32_t N, const std::uint32_t & device_id)
	{
	  dim3 num_blocks;
	  std::int32_t threads_per_block;
	  grid_space::grid<1>::get_launch_params(num_blocks, threads_per_block, N, device_id);
	  std::cout<<"num_blocks.x: "<<num_blocks.x<<" num_blocks.y: "<<num_blocks.y<<" num_blocks.z: "<<num_blocks.z<<" threads_per_block: "<<threads_per_block<<" N:" <<N<<"\n";
	  // call kernel
	  axps<<<num_blocks,threads_per_block>>>(a,x,s,N);
	  return 0;
	  
	}
  template <class Precision_Type>
	HOST std::int32_t call_axps_async(Precision_Type a, Precision_Type * x, Precision_Type s, std::uint32_t N, cudaStream_t & stream, const std::uint32_t & device_id)
	{
	  /*
	  dim3 num_blocks;
	  std::int32_t threads_per_block;
	  const std::uint32_t tile_dim = 16;
	  const std::uint32_t block_row = 8;
	  const std::uint32_t shared_memory = tile_dim * tile_dim * sizeof(Precision_Type);
	  grid<1> one_dimensional_grid;
	  one_dimensional_grid(num_blocks, threads_per_block, N, device_id);
	  std::cout<<"num_blocks.x: "<<num_blocks.x<<" num_blocks.y: "<<num_blocks.y<<" num_blocks.z: "<<num_blocks.z<<" threads_per_block: "<<threads_per_block<<" N:" <<N<<"\n";
	  // call kernel
	  axps<<<num_blocks,threads_per_block, shared_memory, stream>>>(a,x,s,N, tile_dim, block_row);*/
	  return 0;
	  
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
		cudaGetDeviceProperties(&properties, device_id);
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
	    HOST bool get_grid_dim(std::uint32_t N, const std::uint32_t & device_id)
		{
		  std::uint64_t max_outputs_1d_kernel{0};
		  std::uint64_t max_outputs_2d_kernel{0};
		  std::uint64_t max_outputs_3d_kernel{0};
		  cuda_device_properties::max_threads<1>::get_max(max_outputs_1d_kernel,device_id); 
		  cuda_device_properties::max_threads<2>::get_max(max_outputs_2d_kernel,device_id); 
		  cuda_device_properties::max_threads<3>::get_max(max_outputs_3d_kernel,device_id); 
		  if(N <= max_outputs_1d_kernel)
		  {
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
