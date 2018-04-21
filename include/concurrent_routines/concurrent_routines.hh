#ifndef CONCURRENT_ROUTINES_HH
#define CONCURRENT_ROUTINES_HH
#include "macros.hh"
#include <thread>
#include <cstdint>
#define MAX_CPU_THREADS std::thread::hardware_concurrency()
namespace zinhart
{
  //this function is used by each thread to determine what pieces of data it will operate on
  HOST void partition(const std::uint32_t thread_id, const std::uint32_t & n_threads, const std::uint32_t & n_elements, std::uint32_t & start, std::uint32_t & stop);
 
  //CPU WRAPPERS
  HOST void paralell_saxpy_cpu(
  		const float & a, float * x, float * y,
	  	const std::uint32_t & n_elements, const std::uint32_t & n_threads = MAX_CPU_THREADS
	  );
  template<class InputIt, class OutputIt>
  HOST OutputIt paralell_copy_cpu(InputIt first, InputIt last, OutputIt output_it, const std::uint32_t & n_threads = MAX_CPU_THREADS);

  template< class InputIt, class T >
  HOST T paralell_accumulate_cpu( InputIt first, InputIt last, T init,
									const std::uint32_t & n_threads = MAX_CPU_THREADS);



  template < class InputIt, class UnaryFunction >
  HOST UnaryFunction paralell_for_each_cpu(InputIt first, InputIt last, UnaryFunction f,
	  	                              const std::uint32_t & n_threads = MAX_CPU_THREADS  );
  
  template < class InputIt, class OutputIt, class UnaryOperation >
  HOST OutputIt paralell_transform_cpu(InputIt first, InputIt last, OutputIt output_first, UnaryOperation unary_op,
									   const std::uint32_t & n_threads = MAX_CPU_THREADS );
  template < class BidirectionalIt, class Generator >
  HOST void paralell_generate_cpu(BidirectionalIt first, BidirectionalIt last, Generator g,
	   const std::uint32_t & n_threads = MAX_CPU_THREADS);


//#if CUDA_ENABLED == true
  //GPU WRAPPERS
  HOST int parallel_saxpy_gpu(const float & a, float * x, float * y, const std::uint32_t N);
//#endif
}
#include "ext/concurrent_routines_ext.tcc"
#endif
