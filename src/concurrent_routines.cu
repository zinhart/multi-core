#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/timer.hh"
#include <iostream>
namespace zinhart
{
  //EXPLICIT INSTANTIATIONS (to make templates , such as the wrapper functions, in defined in the .cu available in main.cc)

  // KERNELS


  // GPU WRAPPERS
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
