#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_error.hh"
#include "concurrent_routines/timer.hh"
#include <iostream>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
namespace zinhart
{
  //EXPLICIT INSTANTIATIONS (to make templates , such as the wrapper functions, in defined in the .cu available in main.cc)
  
  template HOST std::int32_t call_axps(const float & a, float * x, const float & s, const std::uint32_t & N, const std::uint32_t & device_id);
  template HOST std::int32_t call_axps(const double & a, double * x, const double & s, const std::uint32_t & N, const std::uint32_t & device_id);

  template HOST std::int32_t call_axps_async(const float & a, float * x, const float & s, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);
  template HOST std::int32_t call_axps_async(const double & a, double * x, const double & s, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);


  template HOST std::int32_t reduce(const std::int32_t * in, std::int32_t * out, const std::uint32_t & N, const std::uint32_t & device_id);
  template HOST std::int32_t reduce(const float * in, float * out, const std::uint32_t & N, const std::uint32_t & device_id);
  template HOST std::int32_t reduce(const double * in, double * out, const std::uint32_t & N, const std::uint32_t & device_id);

  template HOST std::int32_t reduce(const std::int32_t * in, std::int32_t * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);
  template HOST std::int32_t reduce(const float * in, float * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);
  template HOST std::int32_t reduce(const double * in, double * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);

  // KERNELS
  
  template <class Precision_Type>
	__global__ void axps(Precision_Type a, Precision_Type * x, Precision_Type s, std::uint32_t N)
	{
	  for(std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;  thread_id < N; thread_id += blockDim.x * gridDim.x)
		x[thread_id] = a * x[thread_id] + s;
	}

  // taken from cuda samples but add thread_fence to get rid of having an output array
  template <class Precision_Type, std::uint32_t block_size, bool n_is_pow2>
	__global__ void reduction_kernel(const Precision_Type * in, Precision_Type * out, const std::uint32_t N)
	{
	  // Handle to thread block group
	  cg::thread_block cta = cg::this_thread_block();
	  extern __shared__ std::uint8_t my_smem[];
	  Precision_Type *sdata = reinterpret_cast<Precision_Type *>(my_smem);

	  // perform first level of reduction,
	  // reading from global memory, writing to shared memory
	  std::uint32_t tid = threadIdx.x;
	  std::uint32_t i = blockIdx.x * block_size * 2 + threadIdx.x;
	  std::uint32_t gridSize = block_size * 2 * gridDim.x;

	  Precision_Type mySum{0};

	  // we reduce multiple elements per thread.  The number is determined by the
	  // number of active thread blocks (via gridDim).  More blocks will result
	  // in a larger gridSize and therefore fewer elements per thread
	  while (i < N)
	  {
		  mySum += in[i];
		  // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		  if (n_is_pow2 || i + block_size < N)
			  mySum += in[i + block_size];
		  i += gridSize;
	  }

	  // each thread puts its local sum into shared memory
	  sdata[tid] = mySum;
	  cg::sync(cta);

	  // do reduction in shared mem
	  if ((block_size >= 512) && (tid < 256)){ sdata[tid] = mySum = mySum + sdata[tid + 256]; }
	  cg::sync(cta);
	  if ((block_size >= 256) &&(tid < 128)){ sdata[tid] = mySum = mySum + sdata[tid + 128]; }
	  cg::sync(cta);
	  if ((block_size >= 128) && (tid <  64)){ sdata[tid] = mySum = mySum + sdata[tid +  64];}
	  cg::sync(cta);
#if (__CUDA_ARCH__ >= 300 )
	  if ( tid < 32 )
	  {
		  cg::coalesced_group active = cg::coalesced_threads();

		  // Fetch final intermediate sum from 2nd warp
		  if (block_size >=  64) mySum += sdata[tid + 32];
		  // Reduce final warp using shuffle
		  for (int offset = warpSize/2; offset > 0; offset /= 2) 
		  {
			   mySum += active.shfl_down(mySum, offset);
		  }
	  }
#else
	  // fully unroll reduction within a single warp
	  if ((block_size >=  64) && (tid < 32)){ sdata[tid] = mySum = mySum + sdata[tid + 32]; }
	  cg::sync(cta);
	  if ((block_size >=  32) && (tid < 16)){ sdata[tid] = mySum = mySum + sdata[tid + 16]; }
	  cg::sync(cta);
	  if ((block_size >=  16) && (tid <  8)){ sdata[tid] = mySum = mySum + sdata[tid +  8]; }
	  cg::sync(cta);
	  if ((block_size >=   8) && (tid <  4)){ sdata[tid] = mySum = mySum + sdata[tid +  4]; }
	  cg::sync(cta);
	  if ((block_size >=   4) && (tid <  2)){ sdata[tid] = mySum = mySum + sdata[tid +  2]; }
	  cg::sync(cta);
	  if ((block_size >=   2) && ( tid <  1)){ sdata[tid] = mySum = mySum + sdata[tid + 1]; }
	  cg::sync(cta);
#endif

	  // write result for this block to global mem
	  if (tid == 0) out[blockIdx.x] = mySum;
	}

  // GPU WRAPPERS
  
  template <class Precision_Type>
	HOST std::int32_t call_axps(const Precision_Type & a, Precision_Type * x, const Precision_Type & s, const std::uint32_t & N, const std::uint32_t & device_id)
	{
	  if(zinhart::check_cuda_api(cudaError_t(cudaSetDevice(device_id)), __FILE__, __LINE__) != 0)
		return 1;
	  dim3 num_blocks;
	  dim3 threads_per_block;
	  grid_space::get_launch_params(num_blocks, threads_per_block, N, device_id);
	  /*std::cout<<"N a:" <<N<<"\n";
	  std::cout<<"num_blocks.x: "<<num_blocks.x<<" num_blocks.y: "<<num_blocks.y<<" num_blocks.z: "<<num_blocks.z<<"\n";
	  std::cout<<"threads_per_block.x: "<<threads_per_block.x<<" threads_per_block.y: "<<threads_per_block.y<<" threads_per_block.z: "<< threads_per_block.z<<"\n";	*/
	  // call kernel
	  axps<<<num_blocks,threads_per_block>>>(a,x,s,N);
	  return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()), __FILE__,__LINE__);
	}
  template <class Precision_Type>
	HOST std::int32_t call_axps_async(const Precision_Type & a, Precision_Type * x, const Precision_Type & s, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id)
	{
	  if(zinhart::check_cuda_api(cudaError_t(cudaSetDevice(device_id)), __FILE__, __LINE__) != 0)
		return 1;
	  dim3 num_blocks;
	  dim3 threads_per_block;
	  std::uint32_t elements_per_thread{0};
	  grid_space::get_launch_params(num_blocks, threads_per_block, N, elements_per_thread, device_id, Precision_Type{});
	 /* std::cout<<"N:" <<N<<"\n";
	  std::cout<<"num_blocks.x: "<<num_blocks.x<<" num_blocks.y: "<<num_blocks.y<<" num_blocks.z: "<<num_blocks.z<<"\n";
	  std::cout<<"threads_per_block.x: "<<threads_per_block.x<<" threads_per_block.y: "<<threads_per_block.y<<" threads_per_block.z: "<< threads_per_block.z<<"\n";	 */
	  axps<<<num_blocks, threads_per_block, 0, stream>>>(a, x, s, N);
	  return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()), __FILE__,__LINE__);
	}

  template <class Precision_Type>
	HOST std::int32_t reduce(const Precision_Type * in, Precision_Type * out, const std::uint32_t & N, const std::uint32_t & device_id)
	{
	  if(zinhart::check_cuda_api(cudaError_t(cudaSetDevice(device_id)), __FILE__, __LINE__) != 0)
		return 1;
	   std::uint32_t max_threads{0};
	   std::uint32_t max_shared_memory_per_block{0};
	   cuda_device_properties::get_max_threads_per_block(max_threads, device_id);
	   cuda_device_properties::get_max_shared_memory(max_shared_memory_per_block, device_id);
	   dim3 threads_per_block( (N < max_threads * 2) ? zinhart::serial::next_pow2( (N + 1) / 2 ) : max_threads, 1, 1);
	   dim3 num_blocks( (N + (threads_per_block.x * 2 - 1) ) / (threads_per_block.x * 2), 1, 1);
       // when there is only one warp per block, we need to allocate two warps
       // worth of shared memory so that we don't index shared memory out of bounds
	   std::uint32_t shared_memory_bytes = (threads_per_block.x <= 32) ? 2 * threads_per_block.x * sizeof(Precision_Type) : threads_per_block.x * sizeof(Precision_Type);
	   if ( (N&(N - 1)) == 0 )// is N is a power of 2
	   {
		 switch (threads_per_block.x)
		 {
		   case 512: reduction_kernel<Precision_Type, 512, true><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case 256: reduction_kernel<Precision_Type, 256, true><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case 128: reduction_kernel<Precision_Type, 128, true><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case 64:  reduction_kernel<Precision_Type,  64, true><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case 32:  reduction_kernel<Precision_Type,  32, true><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case 16:  reduction_kernel<Precision_Type,  16, true><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case  8:  reduction_kernel<Precision_Type,   8, true><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case  4:  reduction_kernel<Precision_Type,   4, true><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case  2:  reduction_kernel<Precision_Type,   2, true><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case  1:  reduction_kernel<Precision_Type,   1, true><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		 }
	   }
	   else
	   {
		 switch (threads_per_block.x)
		 {
		   case 512: reduction_kernel<Precision_Type, 512, false><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case 256: reduction_kernel<Precision_Type, 256, false><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case 128: reduction_kernel<Precision_Type, 128, false><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case 64:  reduction_kernel<Precision_Type,  64, false><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case 32:  reduction_kernel<Precision_Type,  32, false><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case 16:  reduction_kernel<Precision_Type,  16, false><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case  8:  reduction_kernel<Precision_Type,   8, false><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case  4:  reduction_kernel<Precision_Type,   4, false><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case  2:  reduction_kernel<Precision_Type,   2, false><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		   case  1:  reduction_kernel<Precision_Type,   1, false><<< num_blocks, threads_per_block, shared_memory_bytes >>>(in, out, N); break;
		 }
	   }
	   return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()), __FILE__,__LINE__);
  	}


  template <class Precision_Type>
	HOST std::int32_t reduce(const Precision_Type * in, Precision_Type * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id)
	{
	  if(zinhart::check_cuda_api(cudaError_t(cudaSetDevice(device_id)), __FILE__, __LINE__) != 0)
		return 1;
	   std::uint32_t max_threads{0};
	   std::uint32_t max_shared_memory_per_block{0};
	   cuda_device_properties::get_max_threads_per_block(max_threads, device_id);
	   cuda_device_properties::get_max_shared_memory(max_shared_memory_per_block, device_id);
	   dim3 threads_per_block( (N < max_threads * 2) ? zinhart::serial::next_pow2( (N + 1) / 2 ) : max_threads, 1, 1);
	   dim3 num_blocks( (N + (threads_per_block.x * 2 - 1) ) / (threads_per_block.x * 2), 1, 1);
       // when there is only one warp per block, we need to allocate two warps
       // worth of shared memory so that we don't index shared memory out of bounds
	   std::uint32_t shared_memory_bytes = (threads_per_block.x <= 32) ? 2 * threads_per_block.x * sizeof(Precision_Type) : threads_per_block.x * sizeof(Precision_Type);
	   if ( ( N&(N - 1)) == 0 )// is N is a power of 2
	   {
		 switch (threads_per_block.x)
		 {
		   case 512: reduction_kernel<Precision_Type, 512, true><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case 256: reduction_kernel<Precision_Type, 256, true><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case 128: reduction_kernel<Precision_Type, 128, true><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case 64:  reduction_kernel<Precision_Type,  64, true><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case 32:  reduction_kernel<Precision_Type,  32, true><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case 16:  reduction_kernel<Precision_Type,  16, true><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case  8:  reduction_kernel<Precision_Type,   8, true><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case  4:  reduction_kernel<Precision_Type,   4, true><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case  2:  reduction_kernel<Precision_Type,   2, true><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case  1:  reduction_kernel<Precision_Type,   1, true><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		 }
	   }
	   else
	   {
		 switch (threads_per_block.x)
		 {
		   case 512: reduction_kernel<Precision_Type, 512, false><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case 256: reduction_kernel<Precision_Type, 256, false><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case 128: reduction_kernel<Precision_Type, 128, false><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case 64:  reduction_kernel<Precision_Type,  64, false><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case 32:  reduction_kernel<Precision_Type,  32, false><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case 16:  reduction_kernel<Precision_Type,  16, false><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case  8:  reduction_kernel<Precision_Type,   8, false><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case  4:  reduction_kernel<Precision_Type,   4, false><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case  2:  reduction_kernel<Precision_Type,   2, false><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		   case  1:  reduction_kernel<Precision_Type,   1, false><<< num_blocks, threads_per_block, shared_memory_bytes, stream >>>(in, out, N); break;
		 }
	   }
	   return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()), __FILE__,__LINE__);
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
