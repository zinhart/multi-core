#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_error.hh"
#include "gtest/gtest.h"
#include <random>
#include <limits>
#include <iostream>
#include <list>
#if CUDA_ENABLED == true
#include <cublas_v2.h>
#endif
TEST(gpu_test, gemm_wrapper)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint8_t> uint_dist(1, std::numeric_limits<std::uint8_t>::max() );
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
  std::int32_t A_row = uint_dist(mt);
  std::int32_t A_col = uint_dist(mt);
  std::int32_t A_total_elements = A_row  * A_col;
  std::int32_t B_row = A_col;
  std::int32_t B_col = uint_dist(mt);
  std::int32_t B_total_elements = B_row * B_col;
  std::int32_t C_row = A_row;
  std::int32_t C_col = B_col;
  std::int32_t C_total_elements = C_row * C_col;
  std::cout<<"total pinned memory: "<<std::uint32_t(A_total_elements * B_total_elements * C_total_elements)<<" bytes.\n";
  std::cout<<" ";
  std::int32_t m, n, k, lda, ldb, ldc;
  double * A_host;
  double * B_host;
  double * C_host;
  double * A_host_copy;
  double * B_host_copy;
  double * C_host_copy;
  double * A_device, * B_device, * C_device;

  zinhart::check_cuda_api(cudaHostAlloc((void**)&A_host, A_total_elements * sizeof(double), cudaHostAllocDefault),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaHostAlloc((void**)&B_host, B_total_elements * sizeof(double), cudaHostAllocDefault),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaHostAlloc((void**)&C_host, C_total_elements * sizeof(double), cudaHostAllocDefault),__FILE__,__LINE__);

  zinhart::check_cuda_api(cudaHostAlloc((void**)&A_host_copy, A_total_elements * sizeof(double), cudaHostAllocDefault),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaHostAlloc((void**)&B_host_copy, B_total_elements * sizeof(double), cudaHostAllocDefault),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaHostAlloc((void**)&C_host_copy, C_total_elements * sizeof(double), cudaHostAllocDefault),__FILE__,__LINE__);

  for (std::int32_t i =0; i < A_row; ++i)
  {
	for(std::int32_t j = 0; j < A_col; ++j)
	{
	  A_host[zinhart::serial::idx2r(i,j, A_col)] = real_dist(mt);
	}
  }

  for (std::int32_t i =0; i < B_row; ++i)
  {
	for(std::int32_t j = 0; j < B_col; ++j)
	{
	  B_host[zinhart::serial::idx2r(i,j, B_col)] = real_dist(mt);
	}
  }

  for (std::int32_t i =0; i < C_row; ++i)
  {
	for(std::int32_t j = 0; j < C_col; ++j)
	{
	  C_host[zinhart::serial::idx2r(i,j, C_col)] = 0.0f;
	}
  }

  for(std::int32_t i = 0; i < A_total_elements; ++i)
  {
	A_host_copy[i] = 0;
  }
  for(std::int32_t i = 0; i < B_total_elements; ++i)
  {
	B_host_copy[i] = 0;
  }
  for(std::int32_t i = 0; i < C_total_elements; ++i)
  {
	C_host_copy[i] = 0.0f;
  }

  zinhart::check_cuda_api(cudaMalloc((void**)&A_device,  A_total_elements * sizeof(double)),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaMalloc((void**)&B_device,  B_total_elements * sizeof(double)),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaMalloc((void**)&C_device,  C_total_elements * sizeof(double)),__FILE__,__LINE__);


  zinhart::check_cuda_api(cudaMemcpy(A_device, A_host, A_total_elements * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaMemcpy(B_device, B_host, B_total_elements * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaMemcpy(C_device, C_host, C_total_elements * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__);

  cublasHandle_t context;
  zinhart::check_cublas_api(cublasCreate(&context),__FILE__,__LINE__);
  double alpha = 1;
  double beta = 1; 
  zinhart::serial::gemm_wrapper(m,n,k,lda,ldb,ldc, A_row, A_col, B_row, B_col); 
  //sgemm here
  zinhart::check_cublas_api(cublasDgemm(context, CUBLAS_OP_N, CUBLAS_OP_N,
			  m, n, k,
			  &alpha,
			  B_device, lda,
			  A_device, ldb,
			  &beta,
			  C_device, ldc
			 ),__FILE__,__LINE__);/**/

  
  zinhart::check_cuda_api(cudaMemcpy(A_host_copy, A_device, A_total_elements * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaMemcpy(B_host_copy, B_device, B_total_elements * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaMemcpy(C_host_copy, C_device, C_total_elements * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__);
  zinhart::check_cublas_api(cublasDestroy(context),__FILE__,__LINE__);

  zinhart::serial::serial_matrix_product(A_host, B_host, C_host, A_row, A_col, B_col);
/*  zinhart::print_matrix_row_major(A_host, A_row, A_col,"A_host");
  zinhart::print_matrix_row_major(A_host_copy, A_row, A_col, "A_host_copy");
  zinhart::print_matrix_row_major(B_host, B_row, B_col,"B_host");
  zinhart::print_matrix_row_major(B_host_copy, B_row, B_col, "B_host_copy");
  zinhart::print_matrix_row_major(C_host, C_row, C_col,"C_host");
  zinhart::print_matrix_row_major(C_host_copy, C_row, C_col, "C_host_copy");*/
  double epsilon = .0005;
  for(std::int32_t i = 0; i < A_total_elements; ++i)
  {
	ASSERT_NEAR(A_host[i], A_host_copy[i], epsilon);
  }
  for(std::int32_t i = 0; i < B_total_elements; ++i)
  {
	ASSERT_NEAR(B_host[i], B_host_copy[i], epsilon);
  }
  for(std::int32_t i = 0; i < C_total_elements; ++i)
  {
	ASSERT_NEAR(C_host[i],C_host_copy[i], epsilon)<<"i: "<<i<<"\n";
  }

  zinhart::check_cuda_api(cudaFreeHost(A_host),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaFreeHost(B_host),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaFreeHost(C_host),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaFreeHost(A_host_copy),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaFreeHost(B_host_copy),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaFreeHost(C_host_copy),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaFree(A_device),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaFree(B_device),__FILE__,__LINE__);
  zinhart::check_cuda_api(cudaFree(C_device),__FILE__,__LINE__); 
}

TEST(gpu_test, call_axps)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint32_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max() );
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);

  std::uint32_t N = uint_dist(mt);
  double a = real_dist(mt);
  double s = real_dist(mt);

  double * X_host{nullptr};
  double * X_host_validation{nullptr};
  double * X_device{nullptr};
  std::vector<zinhart::parallel::thread_pool::task_future<void>> serial_axps_results;

  X_host = new double[N];
  X_host_validation = new double[N];

  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc((void**)&X_device,  N * sizeof(double)),__FILE__,__LINE__));

  // initialize
  for(std::uint32_t i = 0; i < N; ++i)
  {
	X_host[i] = real_dist(mt);
	X_host_validation[i] = X_host[i];
  }

  // copy host to device
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(X_device, X_host, N * sizeof(double), cudaMemcpyHostToDevice),__FILE__,__LINE__));

  // call kernel 
  zinhart::call_axps(a, X_device, s,N); 
  // copy device to host
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpy(X_host, X_device, N * sizeof(double), cudaMemcpyDeviceToHost),__FILE__,__LINE__));
  // do serial axps
  for(std::uint32_t i = 0; i < N; ++i)
  {
	serial_axps_results.push_back(zinhart::parallel::default_thread_pool::push_task([](double a, double & x, double s){ x = a * x + s; }, a, std::ref(X_host_validation[i]), s));
  }

  // compare
  for(std::uint32_t i = 0; i < N; ++i)
  { 
    serial_axps_results[i].get();	
	ASSERT_EQ(X_host[i], X_host_validation[i]);
  }
  
  // deallocate
  delete X_host;
  delete X_host_validation;
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(X_device),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__, __LINE__));
}



TEST(gpu_test, call_axps_async)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint32_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max() );
  std::uniform_int_distribution<std::uint8_t> stream_dist(1, 10/*std::numeric_limits<std::uint8_t>::max()*/ );
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);

  std::uint32_t N{uint_dist(mt)};
  std::uint32_t n_streams{stream_dist(mt)};
  double a {real_dist(mt)};
  double s {real_dist(mt)};

  double * X_host{nullptr};
  double * X_host_validation{nullptr};
  double * X_device{nullptr};
  cudaStream_t * stream{nullptr};
  std::list<zinhart::parallel::thread_pool::task_future<void>> serial_axps_results;

  ASSERT_EQ(0, zinhart::check_cuda_api(cudaHostAlloc((void**)&X_host, N * sizeof(double),cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaHostAlloc((void**)&X_host_validation, N * sizeof(double),cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaHostAlloc((void**)&stream, n_streams * sizeof(cudaStream_t),cudaHostAllocDefault),__FILE__,__LINE__));


  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc((void**)&X_device,  N * sizeof(double)),__FILE__,__LINE__));


  // initialize
  for(std::uint32_t i = 0; i < N; ++i)
  {
	X_host[i] = real_dist(mt);
	X_host_validation[i] = X_host[i];
  }

  for (std::uint32_t i = 0; i < n_streams; ++i)
  {
  	// copy host to device
	ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(X_device, X_host, N * sizeof(double), cudaMemcpyHostToDevice, stream[i]),__FILE__,__LINE__));
	// call kernel 
	ASSERT_EQ(0, zinhart::call_axps_async(a, X_device, s, N, stream[i])); 
	// copy device to host
	ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(X_host, X_device, N * sizeof(double), cudaMemcpyDeviceToHost, stream[i]),__FILE__,__LINE__));

	// do serial axps
	for(std::uint32_t j = 0; j < N; ++j)
	{
	  serial_axps_results.push_back(zinhart::parallel::default_thread_pool::push_task([](const double a, double & x, const double s){ x = a * x + s; }, a, std::ref(X_host_validation[j]), s));
	}
	
	cudaStreamSynchronize(stream[i]);
	double epsilon = 0.0005;
	// compare
	for(std::uint32_t j = 0; j < N; ++j)
	{ 	
  	  serial_axps_results.front().get();
	  ASSERT_NEAR(X_host[j], X_host_validation[j], epsilon);
	  serial_axps_results.pop_front();
	}

  }
  
  // deallocate
  cudaFreeHost(X_host);
  cudaFreeHost(X_host_validation);
  cudaFreeHost(stream);
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(X_device),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__, __LINE__));
}
/* commented out b/c not finished
TEST(gpu_test, reduce_sum_async)
{
  std::random_device rd;
  std::mt19937 mt(rd());

  // the default thread pool size is std::hardware_concurrency
  std::uniform_int_distribution<std::uint8_t> stream_dist(1, zinhart::parallel::default_thread_pool::get_default_thread_pool().size());

  // total streams
  const std::uint32_t n_streams{stream_dist(mt)};

  //for any needed random uint, ensure that there are atleast as many data points as streams
  std::uniform_int_distribution<std::uint32_t> uint_dist(n_streams, std::numeric_limits<std::uint16_t>::max() );

  // base vector size
  const std::uint32_t N{uint_dist(mt)};

  //for any needed random real, the distribution will be floats stored in double to reduce floating point error
  std::uniform_real_distribution<float> real_dist(-5.5, 5.5);
    
  // host thread pool
  zinhart::parallel::thread_pool pool(n_streams);
 
  // this is here to determine the size of x_host_out
  std::uint32_t max_threads{0};
  std::uint32_t max_shared_memory_per_block{0};
  zinhart::cuda_device_properties::get_max_threads_per_block(max_threads, 0);
  zinhart::cuda_device_properties::get_max_shared_memory(max_shared_memory_per_block, 0);
  const std::uint32_t threads_per_block{ (N < max_threads * 2) ? zinhart::serial::next_pow2( (N + 1) / 2 ) : max_threads};
  const std::uint32_t num_blocks{ (N + (threads_per_block * 2 - 1) ) / (threads_per_block * 2)};

  // vector sizes adjust for the number of streams / threads
  const std::uint32_t input_vector_lengths = N * n_streams;
  const std::uint32_t output_vector_lengths = num_blocks * n_streams;

  // Host and device pointer declarations 
  double * X_host{nullptr};
  double * X_host_out{nullptr};
  double * X_host_validation{nullptr};
  double * X_device{nullptr};
  double * X_device_out{nullptr};
  
  // Cuda streams
  cudaStream_t * stream{nullptr};

  // Host threads
  std::list<zinhart::parallel::thread_pool::task_future<void>> results;

  // Misc variables: loop counters temp values etc
  std::uint32_t i, j, input_offset, output_offset, device_id;

  // this for the host & summation
  double host_sum[n_streams];
  double device_sum{0.0};

  // Allocate page locked host memory and check for error, each host vector will have it's own workspace
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaHostAlloc((void**)&X_host, input_vector_lengths * sizeof(double),cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaHostAlloc((void**)&X_host_out, output_vector_lengths * sizeof(double),cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaHostAlloc((void**)&X_host_validation, input_vector_lengths * sizeof(double),cudaHostAllocDefault),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaHostAlloc((void**)&stream, n_streams * sizeof(cudaStream_t),cudaHostAllocDefault),__FILE__,__LINE__));

  // Allocate device memory, X_device is same length as X_host & X_host_validation
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc((void**)&X_device, input_vector_lengths * sizeof(double)),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaMalloc((void**)&X_device_out, output_vector_lengths * sizeof(double)),__FILE__,__LINE__));

  // initialize host vectors
  for(i = 0; i < input_vector_lengths; ++i)
  {
  	X_host[i] = real_dist(mt);	
   	X_host_validation[i] = X_host[i];
  }
  for(i = 0; i < output_vector_lengths; ++i)
	X_host_out[i] = 0.0;
  std::cout<<"n_streams: "<<n_streams<<"\n";
  std::cout<<"N: "<<N<<"\n";
  std::cout<<"pool size: "<<pool.size()<<"\n";
  //zinhart::serial::print_matrix_row_major(X_host_validation, N, 1, "X_host");


  // trial loop
  for (i = 0, input_offset = 0, output_offset = 0, device_id = 0; i < n_streams; ++i, input_offset += N, output_offset += num_blocks)
  {
	// copy host memory to device 
	ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(X_device + input_offset, X_host + input_offset, N * sizeof(double), cudaMemcpyHostToDevice, stream[i]),__FILE__,__LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(X_device_out + output_offset, X_host_out + output_offset, num_blocks * sizeof(double), cudaMemcpyHostToDevice, stream[i]),__FILE__,__LINE__));
	
    // call reduction kernel once per stream
	zinhart::reduce(X_device + input_offset, X_device_out + output_offset, N, stream[i], device_id);
	
	// important!
 	host_sum[i] = 0.0; 
    
	// perform serial reduction (summation  algo's here in the interest of accuracy)
	zinhart::parallel::async::neumaier_sum( X_host_validation + input_offset, N, host_sum[i], results, pool);

///	std::cout<<"results_size: "<<results.size()<<"\n";

  }	
  // validation loop
  for (i = 0, input_offset = 0, output_offset = 0; i < n_streams; ++i, input_offset += N, output_offset += num_blocks)
  {
	// synchronize host thread w.r.t each stream to ensure reduction kernels have finished
	cudaStreamSynchronize(stream[i]);

	// copy device memory back to host
	ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(X_host + input_offset, X_device + input_offset, N * sizeof(double), cudaMemcpyDeviceToHost, stream[i]),__FILE__,__LINE__));
	ASSERT_EQ(0, zinhart::check_cuda_api(cudaMemcpyAsync(X_host_out + output_offset, X_device_out + output_offset, num_blocks * sizeof(double), cudaMemcpyDeviceToHost, stream[i]),__FILE__,__LINE__));

	// synchronize host thread w.r.t each stream to ensure each memory copy has finished before validation
	cudaStreamSynchronize(stream[i]);

	// validate the results of an individual stream and host thread pair 
	for(j = input_offset; j < input_offset + N; ++j)
	 ASSERT_DOUBLE_EQ(X_host[i], X_host_validation[i]);

	// sum blocks from gpu
	for(j = output_offset; j < num_blocks; ++j)
	  device_sum += X_host_out[j];

	// synchronize host thread w.r.t to the thread that is the twin of the cuda stream, host_sum will contain a valid result from this line onwards
	for(j = 0; j < n_streams; ++j)
	{
	  results.front().get();
  	  // remove futures from the results list, for the next iteration
	  results.pop_front();
	}
	std::cout<<"num_blocks: "<<num_blocks<<"\n";
	zinhart::serial::print_matrix_row_major(X_host_validation + output_offset, 1, num_blocks, "X_host validation");
	std::cout<<"iteration: "<<i<<" host_sum: "<<host_sum[i]<<" device_sum: "<<device_sum<<"\n";

	//zinhart::serial::print_matrix_row_major(X_host_validation + output_offset, num_blocks, 1, "X_host validation");
	//ASSERT_DOUBLE_EQ(host_sum[i], device_sum)<<"iteration: "<<i<<"\n";
	//ASSERT_DOUBLE_EQ(host_sum[i], X_host_out[0])<<"iteration: "<<i<<"\n";


	// for next iteration
	device_sum = 0.0;
  }
  // Deallocate host memory
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(X_host), __FILE__, __LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(X_host_out), __FILE__, __LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(X_host_validation), __FILE__, __LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFreeHost(stream), __FILE__, __LINE__));

  // Deallocate device memory, and check for errors
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(X_device),__FILE__,__LINE__));
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaFree(X_device_out),__FILE__,__LINE__));
  
  // Reset device and check for errors
  ASSERT_EQ(0, zinhart::check_cuda_api(cudaDeviceReset(),__FILE__, __LINE__));
}
*/
