#include "concurrent_routines/concurrent_routines.hh"
#include <algorithm>
#include <thread>
#include <vector>
namespace zinhart
{ 
  //this function is used by each thread to determine what pieces of data it will operate on
  HOST void map(const std::uint32_t thread_id, const std::uint32_t & n_threads, const std::uint32_t & n_elements, std::uint32_t & start, std::uint32_t & stop)
  {
	//total number of operations that must be performed by each thread
  	const std::uint32_t n_ops = n_elements / n_threads; 
	//may not divide evenly
	const std::uint32_t remaining_ops = n_elements % n_threads;
	//if it's the first thread, start should be 0
	start = (thread_id == 0) ? n_ops * thread_id : n_ops * thread_id + remaining_ops;
	stop = n_ops * (thread_id + 1) + remaining_ops;
  }
/*
 * CPU THREADED ROUTINES
 * */
  HOST void saxpy(const std::uint32_t thread_id,
		  	   const std::uint32_t n_threads, const std::uint32_t n_elements, 
  			   const float a, float * x, float * y 
	  )
  {
	std::uint32_t start = 0, stop = 0;
	map(thread_id, n_threads, n_elements, start, stop);
	//operate on y's elements from start to stop
	for(std::uint32_t op = start; op < stop; ++op)
	{
	  y[op] = a * x[op] + y[op];
	}
  }
/*
 * CPU WRAPPERS IMPLEMENTATION
 * */
  HOST void paralell_saxpy(
		const float & a, float * x, float * y,
		const std::uint32_t & n_elements, const std::uint32_t & n_threads
		)
  { 
	//to identify each thread
	std::uint32_t thread_id = 0;
	std::vector<std::thread> threads(n_threads);
	//initialize each thread
	for(std::thread & t : threads)
	{
	  t = std::thread(saxpy, thread_id, n_threads, n_elements, a, x, y );
	  ++thread_id;
	}
	for(std::thread & t : threads)
	  t.join();
  }
 
  HOST void gemm_wrapper(std::int32_t & m, std::int32_t & n, std::int32_t & k, std::int32_t & lda, std::int32_t & ldb, std::int32_t & ldc, const std::uint32_t LDA, const std::uint32_t SDA, const std::uint32_t LDB, const std::uint32_t SDB)
  {
	m = SDB;
	n = LDA;
	k = LDB;
	lda = m;
	ldb = k;
	ldc = m;
  }
  HOST void geam_wrapper(std::int32_t & m, std::int32_t & n, std::int32_t & k, std::int32_t & lda, std::int32_t & ldb, std::int32_t & ldc, const std::uint32_t LDA, const std::uint32_t SDA, const std::uint32_t LDB, const std::uint32_t SDB)
  {
	m = SDB;
	n = LDA;
	k = LDB;
	lda = m;
	ldb = k;
	ldc = m;
  }
  CUDA_CALLABLE_MEMBER std::uint32_t idx2c(std::int32_t i,std::int32_t j,std::int32_t ld)// for column major ordering, if A is MxN then ld is M
  { return j * ld + i; }
  CUDA_CALLABLE_MEMBER std::uint32_t idx2r(std::int32_t i,std::int32_t j,std::int32_t ld)// for row major ordering, if A is MxN then ld is N
  { return i * ld + j; }
}
