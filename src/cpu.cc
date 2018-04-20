#include "concurrent_routines/concurrent_routines.hh"
//#include "concurrent_routines/concurrent_routines_cpu_ext.hh"
#include <algorithm>
#include <thread>
#include <vector>
namespace zinhart
{ 

/*
 * CPU THREADED ROUTINES
 * */
  void saxpy(const std::uint32_t thread_id,
		  	   const std::uint32_t n_threads, const std::uint32_t n_elements, 
  			   const double a, double * x, double * y 
	  )
  {
	//total number of operations that must be performed by each thread
  	const std::uint32_t n_ops = n_elements / n_threads; 
	//may not divide evenly
	const std::uint32_t remaining_ops = n_elements % n_threads;
	//if it's the first thread, start should be 0
	const std::uint32_t start = (thread_id == 0) ? n_ops * thread_id : n_ops * thread_id + remaining_ops;
	const std::uint32_t stop = n_ops * (thread_id + 1) + remaining_ops;
	//operate on y's elements from start to stop
	for(std::uint32_t op = start; op < stop; ++op)
	{
	  y[op] = a * x[op] + y[op];
	}
  }
  void copy(const double a, double * x, double * y)
  {
  }
/*
 * CPU WRAPPERS
 * */
  void paralell_saxpy_cpu(
		const double & a, double * x, double * y,
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
