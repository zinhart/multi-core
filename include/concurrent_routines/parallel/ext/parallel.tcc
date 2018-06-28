#include <type_traits>
namespace zinhart
{
  namespace parallel
  { 
	namespace async
	{
	  /*
	   * * CPU WRAPPERS IMPLEMENTATION
	   * * */

	  template <class Precision_Type, class Container> 
		HOST void parallel_saxpy(const Precision_Type & a, Precision_Type * x, Precision_Type * y, const std::uint32_t & n_elements, Container & results,
			                     thread_pool & default_thread_pool
	                            )
		{
		  static_assert(std::is_same<typename Container::value_type, zinhart::parallel::thread_pool::task_future<void> >::value, "Container value_type must be zinhart::parallel::thread_pool::task_future<void>\n");
	  	  std::uint32_t thread_id = 0;
		  for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			results.push_back(
				default_thread_pool.add_task(zinhart::parallel::vectorized::saxpy<Precision_Type>, thread_id, n_elements, default_thread_pool.size(), a, x, y)
				);

		}
	  template<class InputIt, class OutputIt, class Container>
		HOST void parallel_copy(const InputIt & first, const InputIt & last, OutputIt & output_first, Container & results, thread_pool & default_thread_pool)
		{
		  static_assert(std::is_same<typename Container::value_type, zinhart::parallel::thread_pool::task_future<void> >::value, "Container value_type must be zinhart::parallel::thread_pool::task_future<void>\n");
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
  			for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			  results.push_back(
			  default_thread_pool.add_task(zinhart::parallel::vectorized::copy<InputIt, OutputIt>, std::cref(first), std::ref(output_first), thread_id, n_elements, default_thread_pool.size() )
			  );
		}
	  template<class InputIt, class OutputIt, class UnaryPredicate, class Container>
		HOST void parallel_copy_if(const InputIt & first, const InputIt & last, OutputIt & output_it, UnaryPredicate pred, Container & results, thread_pool & default_thread_pool)
		{
		  static_assert(std::is_same<typename Container::value_type, zinhart::parallel::thread_pool::task_future<void> >::value, "Container value_type must be zinhart::parallel::thread_pool::task_future<void>\n");
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
  			for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			  results.push_back(
			  default_thread_pool.add_task(zinhart::parallel::vectorized::copy_if<InputIt, OutputIt, UnaryPredicate>, std::cref(first), std::ref(output_it), pred, thread_id, n_elements, default_thread_pool.size() )
			  );
		}
	  template< class ForwardIt, class T, class Container >
		HOST void parallel_replace( ForwardIt first, ForwardIt last, const T & old_value, const T & new_value, Container & results, thread_pool & default_thread_pool )
		{
		  static_assert(std::is_same<typename Container::value_type, zinhart::parallel::thread_pool::task_future<void> >::value, "Container value_type must be zinhart::parallel::thread_pool::task_future<void>\n");

		  //to identify each thread
		  std::uint32_t thread_id = 0;
		  const std::uint32_t n_elements = std::distance(first, last);
		  for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			results.push_back(
			default_thread_pool.add_task(zinhart::parallel::vectorized::replace<ForwardIt, T>, std::ref(first), old_value, new_value, thread_id, n_elements, default_thread_pool.size() )
			);
		}
	  template< class ForwardIt, class UnaryPredicate, class T, class Container >
		HOST void parallel_replace_if( ForwardIt first, ForwardIt last, UnaryPredicate unary_predicate, const T& new_value, Container & results, thread_pool & default_thread_pool )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
		/*	for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			  thread_pool.add_task();
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_replace_if_init<ForwardIt, UnaryPredicate, T>, std::ref(first), std::ref(unary_predicate), std::ref(new_value), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
				t.join();
				*/
		}
	  template< class InputIt, class OutputIt, class T, class Container >
		HOST void parallel_replace_copy( InputIt first, InputIt last, OutputIt output_it, const T & old_value, const T & new_value, Container & results, thread_pool & default_thread_pool )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
	/*	for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			  thread_pool.add_task();
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_replace_copy_init<InputIt, OutputIt, T>, std::ref(first), std::ref(output_it), std::ref(old_value), std::ref(new_value), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
				t.join();
			return output_it;
			*/
		}
	  //new
	  template< class InputIt, class OutputIt, class UnaryPredicate, class T, class Container >
		HOST void parallel_replace_copy_if( InputIt first, InputIt last, OutputIt output_it, UnaryPredicate unary_predicate, const T& new_value, Container & results, thread_pool & default_thread_pool )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
			/*
			for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			  thread_pool.add_task();
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_replace_copy_if_init<InputIt, OutputIt, UnaryPredicate, T>, std::ref(first), std::ref(output_it),std::ref(unary_predicate), std::ref(new_value), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
				t.join();
		  return output_it;
		  */
		}
	  template< class InputIt, class OutputIt, class T, class Container >
		HOST T parallel_inner_product( InputIt first, InputIt last, OutputIt output_it, T value, Container & results, thread_pool & default_thread_pool )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
			/*for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			  thread_pool.add_task();
			
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_inner_product_init<InputIt, OutputIt, T>, std::ref(first), std::ref(output_it), std::ref(value), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
				t.join();

			return value;
			*/
		}
	  template<class InputIt1, class InputIt2, class T, class BinaryOperation1, class BinaryOperation2, class Container>
		HOST T parallel_inner_product( InputIt1 first1, InputIt1 last1, InputIt2 first2, T value, BinaryOperation1 op1,BinaryOperation2 op2, Container & results, thread_pool & default_thread_pool )
		{
			return value;
		}
	  template< class InputIt, class T, class Container >
		HOST T parallel_accumulate( InputIt first, InputIt last, T init, Container & results, thread_pool & default_thread_pool)
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
			/*
			for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			  thread_pool.add_task();
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_accumulate_init<InputIt,T>, std::ref(first), std::ref(init), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
			{
				t.join();
			}
			return init;
			*/
		}
	  template < class InputIt, class UnaryFunction, class Container >
		HOST void parallel_for_each(InputIt first, InputIt last, UnaryFunction f, Container & results, thread_pool & default_thread_pool )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
			/*
			for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			  thread_pool.add_task();
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_for_each_init<InputIt, UnaryFunction>, std::ref(first), std::ref(f), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
			{
				t.join();
			}
			return f;
			*/
		}
	  template < class InputIt, class OutputIt, class UnaryOperation, class Container >
		HOST void parallel_transform(InputIt first, InputIt last, OutputIt output_first, UnaryOperation unary_op, Container & results, thread_pool & default_thread_pool )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
			/*
			for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			  thread_pool.add_task();
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_transform_init<InputIt,OutputIt,UnaryOperation>, std::ref(first), std::ref(output_first), std::ref(unary_op), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
				t.join();
			return output_first;
			*/
		}
	  template < class BidirectionalIt, class Generator, class Container >
		HOST void parallel_generate(BidirectionalIt first, BidirectionalIt last, Generator g, Container & results, thread_pool & default_thread_pool)
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
			/*
			for(thread_id = 0; thread_id < default_thread_pool.size(); ++thread_id)
			  thread_pool.add_task();
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_generate_init<BidirectionalIt, Generator>, std::ref(first), std::ref(g), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
			{
				t.join();
			}
			*/
		}
	}// END NAMESPACE ASYNC
  }// END NAMESPACE PARALLEL
}// END NAMESPACE ZINHART

