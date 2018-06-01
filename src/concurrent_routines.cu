#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/timer.hh"
#include <iostream>
namespace zinhart
{
  //EXPLICIT INSTANTIATIONS (to make templates , such as the wrapper functions, in defined in the .cu available in main.cc)

  template HOST std::int32_t call_axps(double a, double * x, double s, std::uint32_t N, const std::uint32_t & device_id);
  // KERNELS
  
  template <class Precision_Type>
	__global__ void axps(Precision_Type a, Precision_Type * x, Precision_Type s, std::uint32_t N)
	{
	  const std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	  if(thread_id > N )
		return;
	  x[thread_id] = a * x[thread_id] + s;
	}

  // GPU WRAPPERS
  template <class Precision_Type>
	HOST std::int32_t call_axps(Precision_Type a, Precision_Type * x, Precision_Type s, std::uint32_t N, const std::uint32_t & device_id)
	{
	  dim3 num_blocks;
	  std::int32_t threads_per_block;
	  grid<1> one_dimensional_grid;
	  one_dimensional_grid(num_blocks, threads_per_block, N, device_id);
	  std::cout<<"num_blocks.x: "<<num_blocks.x<<" num_blocks.y: "<<num_blocks.y<<" num_blocks.z: "<<num_blocks.z<<" threads_per_block: "<<threads_per_block<<" N:" <<N<<"\n";
	  // call kernel
	  axps<<<num_blocks,threads_per_block>>>(a,x,s,N);
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
		std::uint32_t shared_memory = (threads <= 32) ? 2 * threads * sizeof(Precision_Type) : threads * sizeof(Precision_Type);
	  }

  
}
