#ifndef ZINHART_VECTORIZED_HH
#define ZINHART_VECTORIZED_HH
namespace zinhart
{
  namespace parallel
  {
	namespace vectorized
	{
	  template <class Precision_Type>
		HOST void saxpy(const std::uint32_t thread_id,
					    const std::uint32_t & n_elements, const std::uint32_t & n_threads, 
					    const Precision_Type a, Precision_Type * x, Precision_Type * y)
		{
		  std::uint32_t start = 0, stop = 0;
		  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
		  //operate on y's elements from start to stop
		  for(std::uint32_t op = start; op < stop; ++op)
		  {
			y[op] = a * x[op] + y[op];
		  }
		}
	  template<class InputIt, class OutputIt>
		HOST void parallel_copy_init(InputIt input_it, OutputIt output_it,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			//here stop start is how much we should increment the (output/input)_it
			for(std::uint32_t op = start; op < stop; ++op)
				*(output_it + op) = *(input_it + op);
		}
	  template<class InputIt, class OutputIt, class UnaryPredicate>
		HOST void parallel_copy_if_init(InputIt first, OutputIt output_it, UnaryPredicate pred,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				if(pred( *(first + op) ))
					*(output_it + op) = *(first + op);
		}
	  template< class ForwardIt, class T >
		HOST void parallel_replace_init( ForwardIt first, const T & old_value, const T & new_value, 
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				if(*(first + op) == old_value)
					*(first + op) = new_value;
		}
	  template< class ForwardIt, class UnaryPredicate, class T >
		HOST void parallel_replace_if_init( ForwardIt first, UnaryPredicate unary_predicate, const T & new_value, 
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				if( unary_predicate( *(first + op) ) )
					*(first + op) = new_value;
		}
	  template< class InputIt, class OutputIt, class T >
		HOST void parallel_replace_copy_init( InputIt first, OutputIt output_it, const T & old_value, const T & new_value, 
			const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				*(output_it + op) = (  *(first + op) == old_value) ? new_value : *(first + op);
		}
	  template< class InputIt, class OutputIt, class UnaryPredicate, class T >
		HOST void parallel_replace_copy_if_init( InputIt first, OutputIt output_it, UnaryPredicate pred, const T& new_value,
	  const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				*(output_it + op) = ( pred( *(first + op) ) ) ? new_value : *(first + op);
		}
	  template< class InputIt1, class InputIt2, class T >
		HOST void parallel_inner_product_init( InputIt1 first1, InputIt2 first2, T & value,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			// all threads will contribute to the final value of this memory address
			for(std::uint32_t op = start; op < stop; ++op)
				value = value + *(first1 + op) * *(first2 + op);
		}
	  //new
	  template<class InputIt1, class InputIt2, class T, class BinaryOperation1, class BinaryOperation2>
		HOST void parallel_inner_product_init( InputIt1 first1, InputIt2 first2, T & value, BinaryOperation1 op1, BinaryOperation2 op2,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			// all threads will contribute to the final value of this memory address
			for(std::uint32_t op = start; op < stop; ++op)
				value = op1(value, op2( *(first1 + op) ,  *(first2 + op) ));
		}
	  template< class InputIt, class T >
		HOST void parallel_accumulate_init(InputIt first, T & init,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			// all threads will contribute to the final value of this memory address
			for(std::uint32_t op = start; op < stop; ++op)
			{
				init = init + *(first + op);
			}
		}  
	  template< class InputIt, class UnaryFunction >
		HOST void parallel_for_each_init(InputIt first, UnaryFunction f,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			//call f on each element
			for(std::uint32_t op = start; op < stop; ++op)
			{
				f( *(first + op) );
			}
		}
	  template<class InputIt, class OutputIt, class UnaryOperation>
		HOST void parallel_transform_init(InputIt input_it, OutputIt output_it, UnaryOperation unary_op,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			//same deal as copy really
			for(std::uint32_t op = start; op < stop; ++op)
			{
				*(output_it + op) = unary_op( *(input_it + op) );
			}
		}
	  template< class BidirectionalIt, class Generator >
		HOST void parallel_generate_init(BidirectionalIt first, Generator g,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			//call f on each element
			for(std::uint32_t op = start; op < stop; ++op)
			{
				*(first + op) = g();
			}
		}
	}// END NAMESPACE VECTORIZED
  } // END NAMESPACE PARALLEL
}// END NAMESPACE ZINHART
#endif
