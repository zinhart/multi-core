#include "concurrent_routines/concurrent_routines.hh"
#include <algorithm>
#include <thread>
#include <vector>
namespace zinhart
{ 
  //this function is used by each thread to determine what pieces of data it will operate on
  HOST void partition(const std::uint32_t thread_id, const std::uint32_t & n_threads, const std::uint32_t & n_elements, std::uint32_t & start, std::uint32_t & stop)
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
	partition(thread_id, n_threads, n_elements, start, stop);
	//operate on y's elements from start to stop
	for(std::uint32_t op = start; op < stop; ++op)
	{
	  y[op] = a * x[op] + y[op];
	}
  }
/*
 * CPU WRAPPERS IMPLEMENTATION
 * */
  HOST void paralell_saxpy_cpu(
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
}
