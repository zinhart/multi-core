namespace zinhart
{
  namespace parallel
  { 
	namespace async
	{
	  /*
	   * * CPU WRAPPERS IMPLEMENTATION
	   * * */
	  template<class InputIt, class OutputIt>
		HOST OutputIt paralell_copy(InputIt first, InputIt last, OutputIt output_first, const std::uint32_t & n_threads)
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_copy_init<InputIt,OutputIt>, std::ref(first), std::ref(output_first), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
				t.join();
			return output_first;
		}
	  template<class InputIt, class OutputIt, class UnaryPredicate>
		HOST OutputIt paralell_copy_if(InputIt first, InputIt last, OutputIt output_it, UnaryPredicate pred,const std::uint32_t & n_threads)
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_copy_if_init<InputIt, OutputIt, UnaryPredicate>, std::ref(first), std::ref(output_it), pred, thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
				t.join();
			return output_it;
		}
	  template< class ForwardIt, class T >
		HOST void parallel_replace( ForwardIt first, ForwardIt last, const T & old_value, const T & new_value, const std::uint32_t & n_threads )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_replace_init<ForwardIt, T>, std::ref(first), old_value, std::ref(new_value), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
				t.join();
		}
	  template< class ForwardIt, class UnaryPredicate, class T >
		HOST void parallel_replace_if( ForwardIt first, ForwardIt last, UnaryPredicate unary_predicate, const T& new_value, const std::uint32_t & n_threads )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
			std::vector<std::thread> threads(n_threads);
			//initialize each thread
			for(std::thread & t : threads)
			{
				t = std::thread(zinhart::parallel::vectorized::parallel_replace_if_init<ForwardIt, UnaryPredicate, T>, std::ref(first), std::ref(unary_predicate), std::ref(new_value), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
				t.join();
		}
	  template< class InputIt, class OutputIt, class T >
		HOST OutputIt parallel_replace_copy( InputIt first, InputIt last, OutputIt output_it, const T & old_value, const T & new_value, const std::uint32_t & n_threads )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
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
		}
	  //new
	  template< class InputIt, class OutputIt, class UnaryPredicate, class T >
		HOST OutputIt parallel_replace_copy_if( InputIt first, InputIt last, OutputIt output_it, UnaryPredicate unary_predicate, const T& new_value, const std::uint32_t & n_threads )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
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
		}
	  template< class InputIt, class OutputIt, class T >
		HOST T parallel_inner_product( InputIt first, InputIt last, OutputIt output_it, T value, const std::uint32_t & n_threads )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
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
		}
	  template<class InputIt1, class InputIt2, class T, class BinaryOperation1, class BinaryOperation2>
		HOST T parallel_inner_product( InputIt1 first1, InputIt1 last1, InputIt2 first2, T value, BinaryOperation1 op1,BinaryOperation2 op2, const std::uint32_t & n_threads )
		{
			return value;
		}
	  template< class InputIt, class T >
		HOST T paralell_accumalute( InputIt first, InputIt last, T init,
									const std::uint32_t & n_threads = MAX_CPU_THREADS	  )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
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
		}
	  template < class InputIt, class UnaryFunction >
		HOST UnaryFunction paralell_for_each(InputIt first, InputIt last, UnaryFunction f,const std::uint32_t & n_threads )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
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
		}
	  template < class InputIt, class OutputIt, class UnaryOperation >
		HOST OutputIt paralell_transform(InputIt first, InputIt last, OutputIt output_first, UnaryOperation unary_op, const std::uint32_t & n_threads )
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
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
		}
	  template < class BidirectionalIt, class Generator >
		HOST void paralell_generate(BidirectionalIt first, BidirectionalIt last, Generator g, const std::uint32_t & n_threads)
		{
			//to identify each thread
			std::uint32_t thread_id = 0;
			const std::uint32_t n_elements = std::distance(first, last);
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
		}
	}// END NAMESPACE ASYNC
  }// END NAMESPACE PARALLEL
}// END NAMESPACE ZINHART

