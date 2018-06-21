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

	
  template HOST std::int32_t reduce_async(std::int32_t * in, std::int32_t * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);
  template HOST std::int32_t reduce_async(float * in, float * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);
  template HOST std::int32_t reduce_async(double * in, double * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id);

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
	 // dim3 threads_per_block(1024, 1, 1);
	 // dim3 num_blocks( (N + dimBlock.x - 1) / threads_per_block.x, 1, 1);
	 dim3 num_blocks;
	 dim3 threads_per_block;
	 std::uint32_t shared_memory_bytes = threads_per_block.x * sizeof(Precision_Type);
	 //grid_space::get_launch_params(num_blocks, threads_per_block, N, elements_per_thread, device_id);
	 //reduce_kernel<num_blocks, threads_per_block>();
   	 return zinhart::check_cuda_api(cudaError_t(cudaGetLastError()), __FILE__,__LINE__);
	}


  template <class Precision_Type>
	HOST std::int32_t reduce_async(Precision_Type * in, Precision_Type * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id)
	{
	  if(zinhart::check_cuda_api(cudaError_t(cudaSetDevice(device_id)), __FILE__, __LINE__) != 0)
		return 1;
	 // dim3 threads_per_block(1024, 1, 1);
	 // dim3 num_blocks( (N + dimBlock.x - 1) / threads_per_block.x, 1, 1);
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
