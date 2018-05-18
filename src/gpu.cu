#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/timer.hh"
#include <iostream>
namespace zinhart
{
  //EXPLICIT INSTANTIATIONS (to make templates , such as the wrapper functions, in defined in the .cu available in main.cc)
  template HOST std::int32_t parallell_naive_matrix_product_gpu(float * A, float * B, float * C, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB, std::uint32_t LDC, std::uint32_t SDC );
  template HOST std::int32_t parallell_naive_matrix_product_gpu(double * A, double * B, double * C, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB,	std::uint32_t LDC, std::uint32_t SDC );
  template HOST std::int32_t shared_matrix_product(float * A, float * B, float * C, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB, std::uint32_t LDC, std::uint32_t SDC );
  template HOST std::int32_t shared_matrix_product(double * A, double * B, double * C, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB,	std::uint32_t LDC, std::uint32_t SDC );
  template HOST std::int32_t parallell_naive_matrix_transpose_gpu(float * O, float * I, std::uint32_t LDA, std::uint32_t SDA);
  template HOST std::int32_t parallell_naive_matrix_transpose_gpu(double * O, double * I, std::uint32_t LDA, std::uint32_t SDA);
  template HOST std::int32_t parallell_shared_matrix_transpose_gpu(float * O, float * I, std::uint32_t LDA, std::uint32_t SDA);
  template HOST std::int32_t parallell_shared_matrix_transpose_gpu(double * O, double * I, std::uint32_t LDA, std::uint32_t SDA);
  template HOST std::int32_t dgemm_wrapper(float * A, float * B, float * C, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB);
  template HOST std::int32_t dgemm_wrapper(double * A, double * B, double * C, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB);


  // KERNELS
  __global__ void saxpy_kernel(const float a, float * x, float * y, const std::uint32_t N)
  {
   	std::uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread_id > N)
	  return;
	y[thread_id] = a * x[thread_id] + y[thread_id];
  }

  // works for non-square matrices
  template <class Precision_Type>
   __global__ void naive_matrix_product_kernel(Precision_Type * A, Precision_Type * B, Precision_Type * C, std::uint32_t block_dim, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB,	std::uint32_t LDC, std::uint32_t SDC)
	{
	  float sigma = 0.0f;
	  std::int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	  std::int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
	  for(std::uint32_t i = 0; i < (block_dim + SDA - 1) / block_dim; ++i)
	  {
		for(std::uint32_t j = 0; j < block_dim; ++j)
		{
		  if( (i * block_dim + j < SDA && row < LDA) &&  (i * block_dim + j < LDB && col < SDB))
			sigma += A[row * SDA + i * block_dim + j] * B[ (i * block_dim + j) * SDB + col];
		}
	  }
	  if(row < LDC && col < SDC)
		C[ ( (blockIdx.y * blockDim.y + threadIdx.y) * SDC ) + (blockIdx.x * blockDim.x) + threadIdx.x] = sigma;
	}
  // works with rectangular matrices
   template <class Precision_Type>
	 __global__ void shared__memory_matrix_product_kernel(Precision_Type * A, Precision_Type * B, Precision_Type * C, const std::uint32_t block_dim, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB, std::uint32_t LDC, std::uint32_t SDC)
	{
	   float sigma = 0;
	   std::uint32_t row = blockIdx.y * block_dim + threadIdx.y;
	   std::uint32_t col = blockIdx.x * block_dim + threadIdx.x;
	   extern __shared__ float A_shared[]; // the size here is block_dim by block_dim
	   extern __shared__ float B_shared[]; // the size here is block_dim by block_dim
	   for (std::uint32_t i = 0; i < (block_dim + SDA - 1) / block_dim; ++i) 
	   {
		 if ( i * block_dim + threadIdx.x < SDA && row < LDA)
		    A_shared[ threadIdx.y * block_dim + threadIdx.x ] = A[row * SDA + i * block_dim + threadIdx.x];
		 else
		   A_shared[ threadIdx.y * block_dim + threadIdx.x ] = 0.0;
		
		 if (i * block_dim + threadIdx.y < LDB && col < SDB)
		  B_shared[ threadIdx.y * block_dim + threadIdx.x ] = B[ (i * block_dim + threadIdx.y) * SDB + col];
		 else
		   B_shared[ threadIdx.y * block_dim + threadIdx.x ] = 0.0;
		 __syncthreads();
		 for (std::uint32_t n = 0; n < block_dim; ++n)
		  sigma += A_shared[threadIdx.y * block_dim + n] * B_shared[n * block_dim + threadIdx.x]; 

		 __syncthreads();
	   }
	   if (row < LDC && col < SDC)
		 C[((blockIdx.y * blockDim.y + threadIdx.y)*SDC) +  (blockIdx.x * blockDim.x)+ threadIdx.x] = sigma;
	}
 
  template <class Precision_Type>
	__global__ void naive_transpose_kernel(Precision_Type * O, Precision_Type * I, std::uint32_t block_dim, std::uint32_t block_row, std::uint32_t LDA, std::uint32_t SDA)
	{
	  std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	  std::uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;

	  if(row >= LDA || col >= SDA)
		return;
	  std::uint32_t width = gridDim.x * block_dim;
	  for (std::uint32_t j = 0; j < block_dim; j+=block_row)
		O[row * width + (col + j)] = I[(col+j)* width + row];
	}
 // should work with rectangular matrices 
  template <class Precision_Type>
	__global__ void shared_memory_transpose_kernel(Precision_Type * O, Precision_Type * I, const std::uint32_t block_dim, const std::uint32_t LDA, const std::uint32_t SDA)
	{
	  //square taken from devblog nvidia
	  /*extern __shared__ float tile[];
	  int x = blockIdx.x * block_dim + threadIdx.x;
	  int y = blockIdx.y * block_dim + threadIdx.y;

	  if(x >= LDA || y >= SDA)
		return;
	  int width = gridDim.x * block_dim;
	  for (int j = 0; j < block_dim; j += 8)
		tile[threadIdx.y+j * block_dim + threadIdx.x] = I[(y+j)*width + x];


	  __syncthreads();


	  x = blockIdx.y * block_dim + threadIdx.x;  // transpose block offset

	  y = blockIdx.x * block_dim + threadIdx.y;


	  for (int j = 0; j < block_dim; j += 8)
		O[(y+j)*width + x] = tile[threadIdx.x * block_dim + threadIdx.y + j];
		*/
	  // rectangular mine
	  extern __shared__ float block[];

      // read the matrix tile into shared memory
	
	  std::uint32_t row = blockIdx.x * block_dim + threadIdx.x;
	  std::uint32_t col = blockIdx.y * block_dim + threadIdx.y;
	  if ((row < LDA) && (col < SDA))
	  {
		std::uint32_t index_in = col * LDA + row;
		block[threadIdx.y * block_dim + threadIdx.x] = I[index_in];
	  }
	  __syncthreads();
	  // write the transposed matrix tile to global memory
	  row = blockIdx.y * block_dim + threadIdx.x;
	  col = blockIdx.x * block_dim + threadIdx.y;
	  if ((row < SDA) && (col < LDA))
	  {
		std::uint32_t index_out = col * SDA + row;
		O[index_out] = block[threadIdx.x * block_dim + threadIdx.y];
	  }
	}


  // GPU WRAPPERS
  HOST std::int32_t parallel_saxpy_gpu(
  		const float & a, float * x, float * y, const std::uint32_t N)
  {
	cudaError_t error_id;
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
  	dim3 num_blocks;
	std::int32_t warp_size = properties.warpSize;
	std::int32_t threads_per_block = (N + warp_size -1) / warp_size * warp_size;
	if(threads_per_block > 4 * warp_size)
	  threads_per_block = 4 * warp_size;
	num_blocks.x = (N + threads_per_block - 1) / threads_per_block;// number of blocks
	num_blocks.y = 1;
	num_blocks.z = 1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	saxpy_kernel<<<num_blocks, threads_per_block >>>(a, x, y, N);
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

  template <class Precision_Type>
	HOST std::int32_t parallell_naive_matrix_product_gpu(Precision_Type * A, Precision_Type * B, Precision_Type * C, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB, std::uint32_t LDC, std::uint32_t SDC )
	{
	  cudaError_t error_id;
	  const std::uint32_t block_dim = 16;
	  dim3 threads_per_block(block_dim, block_dim, 1);
	  dim3 num_blocks( (SDC + threads_per_block.x - 1) / threads_per_block.x, (LDC + threads_per_block.y - 1)/ threads_per_block.y);
	  cudaEvent_t start, stop;
	  cudaEventCreate(&start);
	  cudaEventCreate(&stop);
	  cudaEventRecord(start);
	  naive_matrix_product_kernel<Precision_Type><<<num_blocks, threads_per_block>>> (A, B, C, block_dim, LDA, SDA, LDB, SDB, LDC, SDC);
	  cudaDeviceSynchronize();
	  cudaEventRecord(stop);
	  cudaEventSynchronize(stop);
	  error_id = cudaGetLastError();
	  if(error_id != cudaSuccess)
	  {
		std::cerr<<"naive matrix product kernel failed to launch with error: "<<cudaGetErrorString(error_id)<<"\n";
		return 1;
	  }

	  float milliseconds = 0;
	  cudaEventElapsedTime(&milliseconds, start, stop);
	  std::cout<<"naive matrix product on matrices of size "<<LDA <<" by "<< SDA <<" and "<< LDB <<" by "<< SDB <<" gpu time: "<<milliseconds<<"ms\n";
	  return 0;
	}
  
  template <class Precision_Type>
	  HOST std::int32_t shared_matrix_product(Precision_Type * A, Precision_Type * B, Precision_Type * C, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB, std::uint32_t LDC, std::uint32_t SDC)
	  {
		cudaError_t error_id;
		const std::uint32_t block_dim = 16;
		dim3 threads_per_block(block_dim,block_dim);
  		dim3 num_blocks( (SDC + threads_per_block.x - 1) / threads_per_block.x, (LDC + threads_per_block.y - 1) / threads_per_block.y);
		const std::uint32_t shared_memory = (block_dim * block_dim + 1) * sizeof(Precision_Type); // in terms of bytes, and apparently + 1 removes bank conflicts
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		shared__memory_matrix_product_kernel<Precision_Type><<<num_blocks, threads_per_block, shared_memory>>> (A, B, C, block_dim, LDA, SDA, LDB, SDB, LDC, SDC);
		cudaDeviceSynchronize();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		error_id = cudaGetLastError();
		if(error_id != cudaSuccess)
		{
		  std::cerr<<"shared matrix product kernel failed to launch with error: "<<cudaGetErrorString(error_id)<<"\n";
		  return 1;
		}

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
  		std::cout<<"shared matrix product on matrices of size "<<LDA <<" by "<< SDA <<" and "<< LDB <<" by "<< SDB <<" gpu time: "<<milliseconds<<"ms\n";
		return 0;
	  }

	template <class Precision_Type>
	  HOST std::int32_t parallell_naive_matrix_transpose_gpu(Precision_Type * O, Precision_Type * I, std::uint32_t LDA, std::uint32_t SDA)
	  {
		cudaError_t error_id;
		const std::uint32_t block_dim = 16;
		const std::uint32_t block_row = 8;
		
		dim3 threads_per_block(block_dim, block_row);
		dim3 num_blocks(LDA / block_dim, SDA / block_dim);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		naive_transpose_kernel<Precision_Type><<<num_blocks, threads_per_block>>> (O, I, block_dim, block_row,LDA, SDA);
		cudaDeviceSynchronize();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		error_id = cudaGetLastError();
		if(error_id != cudaSuccess)
		{
		  std::cerr<<"naive transpose kernel failed to launch with error: "<<cudaGetErrorString(error_id)<<"\n";
		  return 1;
		}

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
  		std::cout<<"naive matrix transpose on matrix of size "<<LDA <<" by "<< SDA <<" gpu time: "<<milliseconds<<"ms\n";
		return 0;
	  }

	template <class Precision_Type>
	  HOST std::int32_t parallell_shared_matrix_transpose_gpu(Precision_Type * O, Precision_Type * I, std::uint32_t LDA, std::uint32_t SDA)
	  {
		cudaError_t error_id;
		const std::uint32_t block_dim = 16;
		// rectangular
		dim3 threads_per_block(block_dim,block_dim);
		dim3 num_blocks( (LDA % block_dim == 0) ? (LDA / block_dim + 1) : (LDA / block_dim) , (SDA % block_dim == 0) ? (SDA / block_dim + 1) : (SDA / block_dim));
		// square
//		dim3 threads_per_block(block_dim,8);
//		dim3 num_blocks(LDA/block_dim,SDA/block_dim);
		const std::uint32_t shared_memory = (block_dim * block_dim + 1) * sizeof(Precision_Type); // in terms of bytes 
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		shared_memory_transpose_kernel<Precision_Type><<<num_blocks, threads_per_block, shared_memory>>> (O, I, block_dim, LDA, SDA);
		cudaDeviceSynchronize();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		error_id = cudaGetLastError();
		if(error_id != cudaSuccess)
		{
		  std::cerr<<"shared transpose kernel failed to launch with error: "<<cudaGetErrorString(error_id)<<"\n";
		  return 1;
		}

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
  		std::cout<<"shared memory matrix transpose on matrix of size "<<LDA <<" by "<< SDA <<" gpu time: "<<milliseconds<<"ms\n";
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

	// assumed to be row major
	template <class Precision_Type>
	  std::int32_t dgemm_wrapper(Precision_Type * A, Precision_Type * B, Precision_Type * C, std::uint32_t LDA, std::uint32_t SDA, std::uint32_t LDB, std::uint32_t SDB)
	  {
	  }
}
