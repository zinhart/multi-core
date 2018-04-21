#ifndef CONCURRENT_ROUTINES_HH
#define CONCURRENT_ROUTINES_HH
#include "macros.hh"
#include <thread>
#include <cstdint>
#define MAX_CPU_THREADS std::thread::hardware_concurrency()
namespace zinhart
{

  //CPU WRAPPERS
  void paralell_saxpy_cpu(
  		const float & a, float * x, float * y,
	  	const std::uint32_t & n_elements, const std::uint32_t & n_threads = MAX_CPU_THREADS
	  );
  template<class InputIt, class OutputIt>
  OutputIt paralell_copy_cpu(InputIt first, InputIt last, OutputIt output_it, const std::uint32_t & n_threads = MAX_CPU_THREADS);

  template< class InputIt, class T >
  T paralell_accumulate_cpu( InputIt first, InputIt last, T init,
									const std::uint32_t & n_threads = MAX_CPU_THREADS);



  template < class InputIt, class UnaryFunction >
  UnaryFunction paralell_for_each_cpu(InputIt first, InputIt last, UnaryFunction f,
	  	                              const std::uint32_t & n_threads = MAX_CPU_THREADS  );
  
  template < class InputIt, class OutputIt, class UnaryOperation >
  OutputIt paralell_transform_cpu(InputIt first, InputIt last, OutputIt output_first, UnaryOperation unary_op,
									   const std::uint32_t & n_threads = MAX_CPU_THREADS );
  template < class BidirectionalIt, class Generator >
  void paralell_generate_cpu(BidirectionalIt first, BidirectionalIt last, Generator g,
	   const std::uint32_t & n_threads = MAX_CPU_THREADS);


#if CUDA_ENABLED == true
  //GPU WRAPPERS
  void launch_gpu_threaded_saxpy(
		const double & a, double * x, double * y,
		const std::uint32_t & n_elements 
	  );
#endif
}
#include "ext/concurrent_routines_ext.tcc"
#endif
