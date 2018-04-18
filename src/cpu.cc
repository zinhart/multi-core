#include "concurrent_routines/concurrent_routines.hh"
#include <algorithm>
#include <thread>
#define MAX_CONCURRENT_OPS std::thread::hardware_concurrency();
void saxpy(const std::uint32_t n_elements, double a, double * x, double * y)
{
	for(std::uint32_t i = 0; i < n_elements; ++i)
	{
		y[i] = a * x[i] + y[i];
	}
}
void launch_cpu_threaded_saxpy(
		const std::uint32_t n_elements, const std::uint32_t n_threads,
		const double a, double * x, double * y
		)
{
	//to identify each thread
	std::uint32_t thread_id = 0;
	std::thread threads[n_threads];
	//initialize each thread
	std::for_each(threads, threads + n_threads,
		      [&thread_id, &threads](std::thread & a_thread)
		      {
			threads[thread_id] = std::thread(cpu_threaded_saxpy, thread_id, n_threads, n_elements, a, x, y ) 
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
		     );
	/*for(thread_id = 0; thread_id < n_threads; ++thread_id)
	{
		threads[thread_id] = std::thread(cpu_threaded_saxpy, thread_id, n_threads, n_elements, a, x, y );
	}
	//for each thread wait until the thread returns
	for(thread_id = 0; thread_id < n_threads; ++thread_id)
	{
		threads[thread_id].join();
	}*/
}
void cpu_threaded_saxpy(
		const std::uint32_t thread_id,
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
