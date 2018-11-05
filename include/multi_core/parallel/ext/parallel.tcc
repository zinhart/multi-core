#include <type_traits>
namespace zinhart
{
  namespace multi_core
  { 
	namespace async
	{
	  template <class precision_type> 
		HOST void saxpy(const precision_type & a, precision_type * x, precision_type * y, const std::uint32_t n_elements, const std::uint32_t n_threads, const std::uint32_t thread_id)
		{
	  	  std::uint32_t  start{0}, stop{0}, op{0};
		  zinhart::multi_core::map(thread_id, n_threads, n_elements, start, stop);
		  for(op = start; op < stop; ++op)
			y[op] = a * x[op] + y[op];
		}
	  /*
	  template<class InputIt, class OutputIt, class container>
		HOST void copy(const InputIt & first, const InputIt & last, OutputIt & output_first, container & results, thread_pool::pool & thread_pool)
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::copy<InputIt, OutputIt>, std::cref(first), std::ref(output_first), thread_id, n_elements, thread_pool.size() )
			);
		}
	  template<class InputIt, class OutputIt, class UnaryPredicate, class container>
		HOST void copy_if(const InputIt & first, const InputIt & last, OutputIt & output_it, UnaryPredicate pred, container & results, thread_pool::pool & thread_pool)
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::copy_if<InputIt, OutputIt, UnaryPredicate>, std::cref(first), std::ref(output_it), pred, thread_id, n_elements, thread_pool.size() )
			);
		}
	  template< class ForwardIt, class T, class container >
		HOST void replace(ForwardIt & first, const ForwardIt & last, const T & old_value, const T & new_value, container & results, thread_pool::pool & thread_pool )
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::replace<ForwardIt, T>, std::ref(first), old_value, new_value, thread_id, n_elements, thread_pool.size() )
			);
		}
	  template< class ForwardIt, class UnaryPredicate, class T, class container >
		HOST void replace_if( ForwardIt & first, const ForwardIt & last, UnaryPredicate unary_predicate, const T& new_value, container & results, thread_pool::pool & thread_pool )
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::replace_if<ForwardIt, UnaryPredicate, T>,std::ref(first), std::ref(unary_predicate), std::ref(new_value), thread_id, n_elements, thread_pool.size() )
			);
		}
	  template< class InputIt, class OutputIt, class T, class container >
		HOST void replace_copy( InputIt & first, const InputIt & last, OutputIt & output_it, const T & old_value, const T & new_value, container & results, thread_pool::pool & thread_pool )
		{

		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::replace_copy<InputIt, OutputIt, T>, std::ref(first), std::ref(output_it), std::ref(old_value), std::ref(new_value), thread_id, n_elements, thread_pool.size() )
			);
		}
	  //new
	  template< class InputIt, class OutputIt, class UnaryPredicate, class T, class container >
		HOST void replace_copy_if( const InputIt & first, const InputIt & last, OutputIt & output_it, UnaryPredicate unary_predicate, const T& new_value, container & results, thread_pool::pool & thread_pool )
		{

		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::replace_copy_if<InputIt, OutputIt, UnaryPredicate, T>, std::ref(first), std::ref(output_it),std::ref(unary_predicate), std::ref(new_value), thread_id, n_elements, thread_pool.size() )
			);
		}
	  template< class InputIt1, class InputIt2, class T, class container>
		HOST void inner_product( const InputIt1 & first, const InputIt1 & last, const InputIt2 & output_it, T & value, container & results, thread_pool::pool & thread_pool )
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::inner_product<InputIt1, InputIt2, T>, std::ref(first), std::ref(output_it), std::ref(value), thread_id, n_elements, thread_pool.size()  )
			);
		}
	  template<class InputIt1, class InputIt2, class T, class BinaryOperation1, class BinaryOperation2, class container>
		HOST void inner_product( const InputIt1 & first1, const InputIt1 & last1, const InputIt2 & first2, T & value, BinaryOperation1 op1,BinaryOperation2 op2, container & results, thread_pool::pool & thread_pool )
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first1, last1);
	  	  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			  results.push_back(
			  thread_pool.add_task(zinhart::multi_core::vectorized::inner_product<InputIt1, InputIt2, T, BinaryOperation1, BinaryOperation2>, std::ref(first1), std::ref(first2), std::ref(value), op1, op2, thread_id, n_elements, thread_pool.size()  )
			  );
		}
	  template< class InputIt, class T, class container >
		HOST void accumulate( const InputIt & first, const InputIt & last, T & init, container & results, thread_pool::pool & thread_pool)
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::accumulate<InputIt, T>, std::ref(first), std::ref(init), thread_id, n_elements, thread_pool.size())
			);
		}
	  template < class InputIt, class UnaryFunction, class container >
		HOST void for_each(const InputIt & first, const InputIt & last, UnaryFunction f, container & results, thread_pool::pool & thread_pool )
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::for_each<InputIt, UnaryFunction>, std::ref(first), std::ref(f), thread_id, n_elements, thread_pool.size())
			);
		}
	  template < class InputIt, class OutputIt, class UnaryOperation, class container >
		HOST void transform(const InputIt & first, const InputIt & last, OutputIt & output_first, UnaryOperation unary_op, container & results, thread_pool::pool & thread_pool )
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::transform<InputIt, OutputIt, UnaryOperation>,  std::ref(first), std::ref(output_first), std::ref(unary_op), thread_id, n_elements, thread_pool.size() )
			);
		}
	  template < class BidirectionalIt, class Generator, class container >
		HOST void generate(const BidirectionalIt & first, const BidirectionalIt & last, Generator g, container & results, thread_pool::pool & thread_pool)
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
	  	  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::generate<BidirectionalIt, Generator>, std::ref(first), std::ref(g), thread_id, n_elements, thread_pool.size())
			);
		}

	  template <class precision_type, class container>
		HOST void kahan_sum(const precision_type * data, const std::uint32_t & data_size, precision_type & global_sum, container & results, thread_pool::pool & thread_pool)
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
	  	  //to identify each thread
		  std::uint32_t thread_id = 0;
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::kahan_sum<precision_type>, data, std::ref(global_sum), thread_id, data_size, thread_pool.size())
			);
		}
	  template <class precision_type, class container>
		HOST void neumaier_sum(const precision_type * data, const std::uint32_t & data_size, precision_type & global_sum, container & results, thread_pool::pool & thread_pool)
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
	  	  //to identify each thread
		  std::uint32_t thread_id = 0;
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::neumaier_sum<precision_type>, data, std::ref(global_sum), thread_id, data_size, thread_pool.size())
			);
		}
  	  template <class precision_type, class container, class binary_predicate>
  		HOST void kahan_sum(const precision_type * vec_1, const precision_type * vec_2, const std::uint32_t & data_size, 
							precision_type & global_sum, 
							binary_predicate bp, 
							container & results, thread_pool::pool & thread_pool
						   )
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
	  	  //to identify each thread
		  std::uint32_t thread_id = 0;
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::kahan_sum<precision_type, binary_predicate>, vec_1, vec_2, std::ref(global_sum), bp, thread_id, data_size, thread_pool.size())
			);
		}
  
	  template <class precision_type, class container, class binary_predicate>
  		HOST void neumaier_sum(const precision_type * vec_1, const precision_type * vec_2, const std::uint32_t & data_size, 
							   precision_type & global_sum,
							   binary_predicate bp,
							   container & results, thread_pool::pool & thread_pool
							  )
		{
		  static_assert(std::is_same<typename container::value_type, zinhart::multi_core::thread_pool::tasks::task_future<void> >::value, "container value_type must be zinhart::multi_core::thread_pool::tasks::task_future<void>\n");
	  	  //to identify each thread
		  std::uint32_t thread_id = 0;
		  for(thread_id = 0; thread_id < thread_pool.size(); ++thread_id)
			results.push_back(
			thread_pool.add_task(zinhart::multi_core::vectorized::neumaier_sum<precision_type, binary_predicate>, vec_1, vec_2, std::ref(global_sum), bp, thread_id, data_size, thread_pool.size())
			);
		  
		}
	  */
	}// END NAMESPACE ASYNC
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
