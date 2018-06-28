#ifndef ZINHART_PARALLELL_HH
#define ZINHART_PARALLELL_HH
#include "../serial/serial.hh" // for map 
#include "vectorized/vectorized.hh"
#include "thread_pool.hh"
namespace zinhart
{
  namespace parallel
  {
	/*
	 * Asynchonous cpu methods. Since these are asynchrounous the called is expected to call resulsts[i].get() to ensure all threads have completed their work
	 * */
	namespace async
	{
	  template <class Precision_Type> 
		HOST void parallel_saxpy(const Precision_Type & a, Precision_Type * x, Precision_Type * y,const std::uint32_t & n_elements, 
			                     std::vector<zinhart::parallel::thread_pool::task_future<void>> & results,
		                         thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
	                            );
	  template<class InputIt, class OutputIt>
		HOST OutputIt parallel_copy(InputIt first, InputIt last, OutputIt output_it, 
			                        thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
								   );

	  template<class InputIt, class OutputIt, class UnaryPredicate>
		HOST OutputIt parallel_copy_if(InputIt first, InputIt last, OutputIt output_it, UnaryPredicate pred,
		                               thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
		                              );

	  template< class ForwardIt, class T >
		HOST void parallel_replace(ForwardIt first, ForwardIt last,const T& old_value, const T& new_value,
		                           thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
		                 	      );

	  template< class ForwardIt, class UnaryPredicate, class T >
		HOST void parallel_replace_if(ForwardIt first, ForwardIt last, UnaryPredicate p, const T& new_value, 
		                              thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
									 );

	  template< class InputIt, class OutputIt, class T >
		HOST OutputIt parallel_replace_copy(InputIt first, InputIt last, OutputIt output_it, const T& old_value, const T& new_value,
		                                    thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
										   );

	  template< class InputIt, class OutputIt, class UnaryPredicate, class T >
		HOST OutputIt parallel_replace_copy_if(InputIt first, InputIt last, OutputIt output_it, UnaryPredicate p, const T& new_value, 
		                                       thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
											   );

	  template< class InputIt1, class InputIt2, class T >
		HOST T parallel_inner_product(InputIt1 first1, InputIt1 last1, InputIt2 first2, T value,
		                              thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
									 );
	  template<class InputIt1, class InputIt2, class T, class BinaryOperation1, class BinaryOperation2>
		HOST T parallel_inner_product(InputIt1 first1, InputIt1 last1, InputIt2 first2, T value, BinaryOperation1 op1, BinaryOperation2 op2,
		                              thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
									 );
	 
	template< class InputIt, class T >
	  HOST T parallel_accumulate(InputIt first, InputIt last, T init,
		                         thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
								);

	template < class InputIt, class UnaryFunction >
	  HOST UnaryFunction parallel_for_each(InputIt first, InputIt last, UnaryFunction f,
		                                   thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
		                                  );
	
	template < class InputIt, class OutputIt, class UnaryOperation >
	  HOST OutputIt parallel_transform(InputIt first, InputIt last, OutputIt output_first, UnaryOperation unary_op,
		                               thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
		                              );

	template < class BidirectionalIt, class Generator >
	  HOST void parallel_generate(BidirectionalIt first, BidirectionalIt last, Generator g,
		                          thread_pool & default_thread_pool = zinhart::parallel::default_thread_pool::get_default_thread_pool()
								 );

	template <class Precision_Type>
	  HOST Precision_Type kahan_sum(Precision_Type * in, const std::uint32_t & N);
	
	template <class Precision_Type>
	  HOST Precision_Type neumaier_sum(Precision_Type * in, const std::uint32_t & N);

	// to call the two methods above
	//template <class Precision_Type>
	 // Precision_Type pairwise_sum; 
	}// END NAMESPACE ASYNC
  }// END NAMESPACE PARALLEL
}// END NAMESPACE ZINHART
#include "ext/parallel.tcc"
#endif
