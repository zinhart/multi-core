#ifndef CONCURRENT_ROUTINES_HH
#define CONCURRENT_ROUTINES_HH
#include "macros.hh"
#include "timer.hh"
#include <thread>
#include <cstdint>
#define MAX_CPU_THREADS std::thread::hardware_concurrency()
namespace zinhart
{
  //this function is used by each thread to determine what pieces of data it will operate on
  HOST void partition(const std::uint32_t thread_id, const std::uint32_t & n_threads, const std::uint32_t & n_elements, std::uint32_t & start, std::uint32_t & stop);
 
  //CPU WRAPPERS
  HOST void paralell_saxpy(
  		const float & a, float * x, float * y,
	  	const std::uint32_t & n_elements, const std::uint32_t & n_threads = MAX_CPU_THREADS
	  );
  template<class InputIt, class OutputIt>
		HOST OutputIt paralell_copy(InputIt first, InputIt last, OutputIt output_it, const std::uint32_t & n_threads = MAX_CPU_THREADS);

  template<class InputIt, class OutputIt, class UnaryPredicate>
		HOST OutputIt paralell_copy_if(InputIt first, InputIt last, OutputIt output_it, UnaryPredicate pred, const std::uint32_t & n_threads = MAX_CPU_THREADS);

	template< class ForwardIt, class T >
		HOST void parallel_replace( ForwardIt first, ForwardIt last,const T& old_value, const T& new_value, const std::uint32_t & n_threads = MAX_CPU_THREADS );

	template< class ForwardIt, class UnaryPredicate, class T >
		HOST void parallel_replace_if( ForwardIt first, ForwardIt last, UnaryPredicate p, const T& new_value, const std::uint32_t & n_threads = MAX_CPU_THREADS );

	template< class InputIt, class OutputIt, class T >
		HOST OutputIt parallel_replace_copy( InputIt first, InputIt last, OutputIt output_it, const T& old_value, const T& new_value, const std::uint32_t & n_threads = MAX_CPU_THREADS );

	template< class InputIt, class OutputIt, class UnaryPredicate, class T >
		HOST OutputIt parallel_replace_copy_if( InputIt first, InputIt last, OutputIt output_it, UnaryPredicate p, const T& new_value, const std::uint32_t & n_threads = MAX_CPU_THREADS );

	template< class InputIt1, class InputIt2, class T >
		HOST T parallel_inner_product( InputIt1 first1, InputIt1 last1, InputIt2 first2, T value, const std::uint32_t & n_threads = MAX_CPU_THREADS );

	template<class InputIt1, class InputIt2, class T, class BinaryOperation1, class BinaryOperation2>
		HOST T parallel_inner_product( InputIt1 first1, InputIt1 last1, InputIt2 first2, T value, BinaryOperation1 op1, BinaryOperation2 op2, const std::uint32_t & n_threads = MAX_CPU_THREADS );
	 
	template< class InputIt, class T >
	HOST T paralell_accumulate( InputIt first, InputIt last, T init,
									  const std::uint32_t & n_threads = MAX_CPU_THREADS);

	template < class InputIt, class UnaryFunction >
	HOST UnaryFunction paralell_for_each(InputIt first, InputIt last, UnaryFunction f,
										const std::uint32_t & n_threads = MAX_CPU_THREADS  );
	
	template < class InputIt, class OutputIt, class UnaryOperation >
	HOST OutputIt paralell_transform(InputIt first, InputIt last, OutputIt output_first, UnaryOperation unary_op,
										 const std::uint32_t & n_threads = MAX_CPU_THREADS );
	template < class BidirectionalIt, class Generator >
	HOST void paralell_generate(BidirectionalIt first, BidirectionalIt last, Generator g,
		 const std::uint32_t & n_threads = MAX_CPU_THREADS);


//#if CUDA_ENABLED == true
  //GPU WRAPPERS
  HOST int parallel_saxpy_gpu(const float & a, float * x, float * y, const std::uint32_t N);
//#endif
}
#include "ext/concurrent_routines_ext.tcc"
#endif
