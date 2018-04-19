#include "concurrent_routines/concurrent_routines.hh"
#include "concurrent_routines/concurrent_routines_cpu_ext.hh"
#include <algorithm>
#include <thread>
#include <vector>
namespace zinhart
{ 
  void launch_cpu_threaded_saxpy(
		const double a, double * x, double * y,
		const std::uint32_t n_elements, const std::uint32_t n_threads
	
		)
  { 
	//to identify each thread
	std::uint32_t thread_id = 0;
	std::vector<std::thread> threads(n_threads);
	//std::thread threads[n_threads];
	//initialize each thread
/*	std::for_each(threads, threads + n_threads,
		      [&threads, &thread_id, &n_threads, &n_elements, &a, &x, &y](std::thread & a_thread)
		      {
			threads[thread_id] = std::thread(saxpy<EXCECUTION_POLICY::PARALLEL>(), thread_id, n_threads, n_elements, a, x, y );
			++thread_id;
		      }
		     );
	//reset
	thread_id = 0;
	//for each thread wait until the thread returns
	std::for_each(threads, threads + n_threads,
		      [&thread_id, &threads](std::thread & a_thread)
		      {
			threads[thread_id].join();
			++thread_id;
		      }
		     );*/
	for(thread_id = 0; thread_id < n_threads; ++thread_id)
	{
//		threads[thread_id] = std::thread(saxpy<EXCECUTION_POLICY::PARALLEL>(), thread_id, n_threads, n_elements, a, x, y );
	}
	//for each thread wait until the thread returns
	for(thread_id = 0; thread_id < n_threads; ++thread_id)
	{
//		threads[thread_id].join();
	}
  }
}
