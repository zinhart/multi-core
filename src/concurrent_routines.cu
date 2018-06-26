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


  template HOST std::int32_t reduce(std::int32_t * in, std::int32_t * out, const std::uint32_t & N, const std::uint32_t & device_id);
  template HOST std::int32_t reduce(float * in, float * out, const std::uint32_t & N, const std::uint32_t & device_id);
  template HOST std::int32_t reduce(double * in, double * out, const std::uint32_t & N, const std::uint32_t & device_id);
  template HOST std::int32_t reduce(std::int32_t * in, std::int32_t * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);
  template HOST std::int32_t reduce(float * in, float * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);
  template HOST std::int32_t reduce(double * in, double * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);

  // KERNELS
  
  template <class Precision_Type>
	__global__ void axps(Precision_Type a, Precision_Type * x, Precision_Type s, std::uint32_t N)
	{
	  for(std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;  thread_id < N; thread_id += blockDim.x * gridDim.x)
		x[thread_id] = a * x[thread_id] + s;
	}

  // ripped from mark harris and my ams148 prof steven reeves( if you're reading this steven I learned alot in your class)
  template <class Precision_Type, std::uint32_t Block_Size>
	__global__ void reduce_kernel(Precision_Type * in, Precision_Type * out, std::uint32_t N)
	{
	  extern __shared__ Precision_Type sdata[];
	  int myId = threadIdx.x + (blockDim.x * 2) * blockIdx.x;
	  int tid = threadIdx.x;
	  int gridSize = blockDim.x * 2 * gridDim.x; 
	  sdata[tid] = 0; 

	  //load shared mem from global mem
	  while(myId < N) { sdata[tid] += in[myId] + in[myId + blockDim.x]; myId += gridSize; }
	  __syncthreads(); 
	  
	  //do reduction over shared memory
	  if(Block_Size >= 512){ if(tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();}
	  if(Block_Size >= 256){ if(tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();}
	  if(Block_Size >= 128){ if(tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	  if(tid < 32)
	  {	
		if(Block_Size >= 64) sdata[tid] += sdata[tid+32];
		if(Block_Size >= 32) sdata[tid] += sdata[tid+16];
		if(Block_Size >= 16) sdata[tid] += sdata[tid+8];
		if(Block_Size >= 8) sdata[tid] += sdata[tid+4];
		if(Block_Size >= 4) sdata[tid] += sdata[tid+2];
		if(Block_Size >= 2) sdata[tid] += sdata[tid+1];
	  }
	  
	  //only tid 0 writes out result!
	  if(tid == 0){ out[blockIdx.x] = sdata[0];}
	}
  template <class T, unsigned int blockSize, bool nIsPow2>
	__global__ void reduce(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    cg::sync(cta);


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    cg::sync(cta);

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        cg::coalesced_group active = cg::coalesced_threads();

        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
             mySum += active.shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    cg::sync(cta);

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    cg::sync(cta);

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    cg::sync(cta);

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    cg::sync(cta);

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    cg::sync(cta);

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    cg::sync(cta);
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
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
	HOST std::int32_t reduce(Precision_Type * in, Precision_Type * out, const std::uint32_t & N, const std::uint32_t & device_id)
	{
	  if(zinhart::check_cuda_api(cudaError_t(cudaSetDevice(device_id)), __FILE__, __LINE__) != 0)
		return 1;
	 dim3 num_blocks;
	 dim3 threads_per_block;

//    dim3 dimBlock(threads, 1, 1);
//    dim3 dimGrid(blocks, 1, 1);
     // when there is only one warp per block, we need to allocate two warps
     // worth of shared memory so that we don't index shared memory out of bounds
	 std::uint32_t shared_memory_bytes = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	 if (isPow2(N))
	 {
	   switch (threads)
	   {
		 case 512: reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case 256: reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	  	 case 128: reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case 64:  reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case 32:  reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;           
	  	 case 16:  reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;           
		 case  8:  reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case  4:  reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case  2:  reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;           
	  	 case  1:  reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);  break;
                
	   } 
	 }
	 else       
 	 {        
	   switch (threads)      
	   {           
		 case 512: reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case 256: reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case 128: reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case 64:  reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case 32:  reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case 16:  reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case  8:  reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case  4:  reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case  2:  reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		 case  1:  reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	   }
	 }


	 break;
	}

	 //grid_space::get_launch_params(num_blocks, threads_per_block, N, elements_per_thread, device_id);
	 //reduce_kernel<num_blocks, threads_per_block>();
   	 return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()), __FILE__,__LINE__);	 
  }


  template <class Precision_Type>
	HOST std::int32_t reduce(Precision_Type * in, Precision_Type * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id)
	{
	  if(zinhart::check_cuda_api(cudaError_t(cudaSetDevice(device_id)), __FILE__, __LINE__) != 0)
		return 1;
	 // dim3 threads_per_block(1024, 1, 1);
	 dim3 num_blocks( (N + dimBlock.x - 1) / threads_per_block.x, 1, 1);
	 dim3 num_blocks;
	 dim3 threads_per_block;
	 std::uint32_t shared_memory_bytes = threads_per_block.x * sizeof(Precision_Type);
	 //grid_space::get_launch_params(num_blocks, threads_per_block, N, elements_per_thread, device_id);
	 //reduce_kernel<num_blocks, threads_per_block>();
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
