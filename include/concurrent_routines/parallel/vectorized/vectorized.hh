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
		HOST void copy(InputIt input_it, OutputIt output_it,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			//here stop start is how much we should increment the (output/input)_it
			for(std::uint32_t op = start; op < stop; ++op)
				*(output_it + op) = *(input_it + op);
		}
	  template<class InputIt, class OutputIt, class UnaryPredicate>
		HOST void copy_if(InputIt first, OutputIt output_it, UnaryPredicate pred,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				if(pred( *(first + op) ))
					*(output_it + op) = *(first + op);
		}
	  template< class ForwardIt, class T >
		HOST void replace( ForwardIt first, const T & old_value, const T & new_value, 
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				if(*(first + op) == old_value)
					*(first + op) = new_value;
		}
	  template< class ForwardIt, class UnaryPredicate, class T >
		HOST void replace_if( ForwardIt first, UnaryPredicate unary_predicate, const T & new_value, 
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				if( unary_predicate( *(first + op) ) )
					*(first + op) = new_value;
		}
	  template< class InputIt, class OutputIt, class T >
		HOST void replace_copy( InputIt first, OutputIt output_it, const T & old_value, const T & new_value, 
			const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				*(output_it + op) = (  *(first + op) == old_value) ? new_value : *(first + op);
		}
	  template< class InputIt, class OutputIt, class UnaryPredicate, class T >
		HOST void replace_copy_if( InputIt first, OutputIt output_it, UnaryPredicate pred, const T& new_value,
	  const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				*(output_it + op) = ( pred( *(first + op) ) ) ? new_value : *(first + op);
		}
	  template< class InputIt1, class InputIt2, class T >
		HOST void inner_product( InputIt1 first1, InputIt2 first2, T & value,
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
		HOST void inner_product( InputIt1 first1, InputIt2 first2, T & value, BinaryOperation1 op1, BinaryOperation2 op2,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			// all threads will contribute to the final value of this memory address
			for(std::uint32_t op = start; op < stop; ++op)
				value = op1(value, op2( *(first1 + op) ,  *(first2 + op) ));
		}
	  template< class InputIt, class T >
		HOST void accumulate(InputIt first, T & init,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			// all threads will contribute to the final value of this memory address
			for(std::uint32_t op = start; op < stop; ++op)
				init = init + *(first + op);
		}  
	  template< class InputIt, class UnaryFunction >
		HOST void for_each(InputIt first, UnaryFunction f,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			//call f on each element
			for(std::uint32_t op = start; op < stop; ++op)
				f( *(first + op) );
		}
	  template<class InputIt, class OutputIt, class UnaryOperation>
		HOST void transform(InputIt input_it, OutputIt output_it, UnaryOperation unary_op,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			//same deal as copy really
			for(std::uint32_t op = start; op < stop; ++op)
				*(output_it + op) = unary_op( *(input_it + op) );
		}
	  template< class BidirectionalIt, class Generator >
		HOST void generate(BidirectionalIt first, Generator g,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
			//call f on each element
			for(std::uint32_t op = start; op < stop; ++op)
				*(first + op) = g();
		}
	  template <class Precision_Type>
		HOST void kahan_sum(const Precision_Type * data, Precision_Type & global_sum, const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
		  std::uint32_t start{0}, stop{0}, op{0};
		  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
		  Precision_Type local_sum{ data[start] };
		  // a running compensation for lost lower-order bits
		  Precision_Type compensation{0.0};
		  for(op = start + 1; op < stop; ++op)
		  {
			Precision_Type y{ data[op] - compensation };
			// lower order bits are lost here with this addition
			Precision_Type t{ local_sum + y };
			// (t - local_sum) cancels the higher order part of y and subtracting y recorvers the low part of y
			compensation = (t - local_sum) - y;
			local_sum = t;
		  }
		  global_sum += local_sum;
		}
	  template <class Precision_Type>
		HOST void neumaier_sum(Precision_Type * data, Precision_Type & global_sum, const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
		  std::uint32_t start{0}, stop{0}, op{0};
		  zinhart::serial::map(thread_id, n_threads, n_elements, start, stop);
		  Precision_Type local_sum{ data[start] };
		  // a running compensation for lost lower-order bits
		  Precision_Type compensation{0.0};
		  for(op = start + 1; op < stop; ++op)
		  {
			Precision_Type t{ local_sum + data[op] };
			if(std::abs(local_sum) >= std::abs(data[op]))
			  // if the local_sum is bigger lower order digitis of in[op] are lost
			  compensation += (local_sum - t) + data[op];
			else
			  // if the local_sum is smaller lower order digits of local_sum are lost
			  compensation += (data[op] - t) + local_sum;
			local_sum = t;
		  }
		  // Correction is applied once
		  global_sum += (local_sum + compensation);
		}
	}// END NAMESPACE VECTORIZED
  } // END NAMESPACE PARALLEL
}// END NAMESPACE ZINHART
#endif
