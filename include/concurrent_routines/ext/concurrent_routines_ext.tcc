#include "../concurrent_routines.hh"
#include <vector>
namespace zinhart
{
/*
 * CPU THREADED ROUTINES
 * */
  template<class InputIt, class OutputIt>
		HOST void parallel_copy_init(InputIt input_it, OutputIt output_it,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			map(thread_id, n_threads, n_elements, start, stop);
			//here stop start is how much we should increment the (output/input)_it
			for(std::uint32_t op = start; op < stop; ++op)
				*(output_it + op) = *(input_it + op);
		}
  template<class InputIt, class OutputIt, class UnaryPredicate>
		HOST void parallel_copy_if_init(InputIt first, OutputIt output_it, UnaryPredicate pred,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				if(pred( *(first + op) ))
					*(output_it + op) = *(first + op);
		}
	template< class ForwardIt, class T >
		HOST void parallel_replace_init( ForwardIt first, const T & old_value, const T & new_value, 
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				if(*(first + op) == old_value)
					*(first + op) = new_value;
		}
	template< class ForwardIt, class UnaryPredicate, class T >
		HOST void parallel_replace_if_init( ForwardIt first, UnaryPredicate unary_predicate, const T & new_value, 
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				if( unary_predicate( *(first + op) ) )
					*(first + op) = new_value;
		}
	template< class InputIt, class OutputIt, class T >
		HOST void parallel_replace_copy_init( InputIt first, OutputIt output_it, const T & old_value, const T & new_value, 
			const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				*(output_it + op) = (  *(first + op) == old_value) ? new_value : *(first + op);
		}
	template< class InputIt, class OutputIt, class UnaryPredicate, class T >
		HOST void parallel_replace_copy_if_init( InputIt first, OutputIt output_it, UnaryPredicate pred, const T& new_value,
	  const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			map(thread_id, n_threads, n_elements, start, stop);
			for(std::uint32_t op = start; op < stop; ++op)
				*(output_it + op) = ( pred( *(first + op) ) ) ? new_value : *(first + op);
		}
	template< class InputIt1, class InputIt2, class T >
		HOST void parallel_inner_product_init( InputIt1 first1, InputIt2 first2, T & value,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads )
		{
			std::uint32_t start = 0, stop = 0;
			map(thread_id, n_threads, n_elements, start, stop);
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
			map(thread_id, n_threads, n_elements, start, stop);
			// all threads will contribute to the final value of this memory address
			for(std::uint32_t op = start; op < stop; ++op)
				value = op1(value, op2( *(first1 + op) ,  *(first2 + op) ));
		}
 
  template< class InputIt, class T >
		HOST void parallel_accumulate_init(InputIt first, T & init,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
		{
			std::uint32_t start = 0, stop = 0;
			map(thread_id, n_threads, n_elements, start, stop);
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
			map(thread_id, n_threads, n_elements, start, stop);
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
			map(thread_id, n_threads, n_elements, start, stop);
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
			map(thread_id, n_threads, n_elements, start, stop);
			//call f on each element
			for(std::uint32_t op = start; op < stop; ++op)
			{
				*(first + op) = g();
			}
		}

/*
 * CPU WRAPPERS IMPLEMENTATION
 * */
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
				t = std::thread(parallel_copy_init<InputIt,OutputIt>, std::ref(first), std::ref(output_first), thread_id, n_elements, n_threads );
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
				t = std::thread(parallel_copy_if_init<InputIt, OutputIt, UnaryPredicate>, std::ref(first), std::ref(output_it), pred, thread_id, n_elements, n_threads );
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
				t = std::thread(parallel_replace_init<ForwardIt, T>, std::ref(first), old_value, std::ref(new_value), thread_id, n_elements, n_threads );
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
				t = std::thread(parallel_replace_if_init<ForwardIt, UnaryPredicate, T>, std::ref(first), std::ref(unary_predicate), std::ref(new_value), thread_id, n_elements, n_threads );
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
				t = std::thread(parallel_replace_copy_init<InputIt, OutputIt, T>, std::ref(first), std::ref(output_it), std::ref(old_value), std::ref(new_value), thread_id, n_elements, n_threads );
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
				t = std::thread(parallel_replace_copy_if_init<InputIt, OutputIt, UnaryPredicate, T>, std::ref(first), std::ref(output_it),std::ref(unary_predicate), std::ref(new_value), thread_id, n_elements, n_threads );
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
				t = std::thread(parallel_inner_product_init<InputIt, OutputIt, T>, std::ref(first), std::ref(output_it), std::ref(value), thread_id, n_elements, n_threads );
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
				t = std::thread(parallel_accumulate_init<InputIt,T>, std::ref(first), std::ref(init), thread_id, n_elements, n_threads );
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
				t = std::thread(parallel_for_each_init<InputIt, UnaryFunction>, std::ref(first), std::ref(f), thread_id, n_elements, n_threads );
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
				t = std::thread(parallel_transform_init<InputIt,OutputIt,UnaryOperation>, std::ref(first), std::ref(output_first), std::ref(unary_op), thread_id, n_elements, n_threads );
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
				t = std::thread(parallel_generate_init<BidirectionalIt, Generator>, std::ref(first), std::ref(g), thread_id, n_elements, n_threads );
				++thread_id;
			}
			for(std::thread & t : threads)
			{
				t.join();
			}
		}
 /*
 * CPU WRAPPERS HELPER FUNCTIONS
 * */
  template<class Precision_Type>
	  HOST void serial_matrix_product(const Precision_Type * A, const Precision_Type * B, Precision_Type * C, const std::uint32_t M, const std::uint32_t N, const std::uint32_t K)
	  {
		for(std::uint32_t a = 0; a < M; ++a)
		  for(std::uint32_t b = 0; b < K; ++b)
			for(std::uint32_t c = 0; c < N; ++c)
			  C[idx2r(a, b, K)] += A[idx2r(a, c, N)] * B[idx2r(c, b, K)];
	  }
	template<class Precision_Type>
	  HOST void print_matrix_row_major(Precision_Type * mat, std::uint32_t mat_rows, std::uint32_t mat_cols, std::string s)
	  {
		 std::cout<<s<<"\n";
		 for(std::uint32_t i = 0; i < mat_rows; ++i)  
		 {
		   for(std::uint32_t j = 0; j < mat_cols; ++j)
		   {
			 std::cout<<mat[idx2r(i,j,mat_cols)]<<" ";
		   }
		   std::cout<<"\n";
		 }
	  }

	// taken from wikipedia 
https://en.wikipedia.org/wiki/Kahan_summation_algorithm	template <class Precision_Type>
	  HOST Precision_Type kahan_sum(Precision_Type * in, const std::uint32_t & N)
	  {
		Precision_Type sum{0.0};
		// a running compensation for lost lower-order bits
		Precision_Type compensation{0.0}; 		
		for(std::uint32_t i  = 1; i < N; ++i)
		{
		  Precision_Type y{in[i] - compensation};
		  // lower order bits are lost here with this addition
		  Precision_Type t{sum + y}; 
		  // (t - sum) cancels the higher order part of y and subtracting y recorvers the low part of y
		  compensation = (t - sum) - y;		  
		  sum = t;
		}
		return sum;
	  }
	
	// taken from wikipedia, this is an improvement on the algorithm above 
https://en.wikipedia.org/wiki/Kahan_summation_algorithm	template <class Precision_Type>
	template <class Precision_Type>
	  HOST Precision_Type neumaier_sum(Precision_Type * in, const std::uint32_t & N)
	  {
		Precision_Type sum{0.0};
		// a running compensation for lost lower-order bits
		Precision_Type compensation{0.0}; 		
		for(std::uint32_t i  = 2; i < N; ++i)
		{
		  Precision_Type t{sum + in[i]};
		  if(std::abs(sum) >= std::abs(in[i]))
			// if the sum is bigger lower order digitis of in[i] are lost
			compensation += (sum - t) + input[i];
		  else
			// if the sum is smaller lower order digits of sum are lost
			compensation += (input[i] - t) + sum;
		  sum = t;
		}
		// Correction is applied once
		return sum + compensation;
	  }

/*
 * GPU TEMPLATE WRAPPERS
 * */
#if CUDA_ENABLED == true

#endif
}

