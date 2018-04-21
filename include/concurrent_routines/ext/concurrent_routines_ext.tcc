#include "../concurrent_routines.hh"
#include <vector>
#include <future>
namespace zinhart
{

/*
 * CPU THREADED ROUTINES
 * */

  template<class InputIt, class OutputIt>
  void parallel_copy(InputIt input_it, OutputIt output_it,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
  {
	//total number of operations that must be performed by each thread
  	const std::uint32_t n_ops = n_elements / n_threads; 
	//may not divide evenly
	const std::uint32_t remaining_ops = n_elements % n_threads;
	//if it's the first thread, start should be 0
	const std::uint32_t start = (thread_id == 0) ? n_ops * thread_id : n_ops * thread_id + remaining_ops;
	const std::uint32_t stop = n_ops * (thread_id + 1) + remaining_ops;
	//here stop start is how much we should increment the (output/input)_it
	for(std::uint32_t op = start; op < stop; ++op)
	{
	  *(output_it + op) = *(input_it + op);
	}
  }
 
  template< class InputIt, class T >
  void parallel_accumulate(InputIt first, T & init,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
  {
	//total number of operations that must be performed by each thread
  	const std::uint32_t n_ops = n_elements / n_threads; 
	//may not divide evenly
	const std::uint32_t remaining_ops = n_elements % n_threads;
	//if it's the first thread, start should be 0
	const std::uint32_t start = (thread_id == 0) ? n_ops * thread_id : n_ops * thread_id + remaining_ops;
	const std::uint32_t stop = n_ops * (thread_id + 1) + remaining_ops;
	// all threads will contribute to the final value of this memory address
	for(std::uint32_t op = start; op < stop; ++op)
	{
	  init = init + *(first + op);
	}
  }  


  template< class InputIt, class UnaryFunction >
  void parallel_for_each(InputIt first, UnaryFunction f,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
  {
	//total number of operations that must be performed by each thread
  	const std::uint32_t n_ops = n_elements / n_threads; 
	//may not divide evenly
	const std::uint32_t remaining_ops = n_elements % n_threads;
	//if it's the first thread, start should be 0
	const std::uint32_t start = (thread_id == 0) ? n_ops * thread_id : n_ops * thread_id + remaining_ops;
	const std::uint32_t stop = n_ops * (thread_id + 1) + remaining_ops;
	//call f on each element
	for(std::uint32_t op = start; op < stop; ++op)
	{
	  f( *(first + op) );
	}
  }


  template<class InputIt, class OutputIt, class UnaryOperation>
  void parallel_transform(InputIt input_it, OutputIt output_it, UnaryOperation unary_op,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
  {
	//total number of operations that must be performed by each thread
  	const std::uint32_t n_ops = n_elements / n_threads; 
	//may not divide evenly
	const std::uint32_t remaining_ops = n_elements % n_threads;
	//if it's the first thread, start should be 0
	const std::uint32_t start = (thread_id == 0) ? n_ops * thread_id : n_ops * thread_id + remaining_ops;
	const std::uint32_t stop = n_ops * (thread_id + 1) + remaining_ops;
	//same deal as copy really
	for(std::uint32_t op = start; op < stop; ++op)
	{
	  *(output_it + op) = unary_op( *(input_it + op) );
	}
  }


  template< class BidirectionalIt, class Generator >
  void parallel_generate(BidirectionalIt first, Generator g,
		const std::uint32_t & thread_id, const std::uint32_t & n_elements, const std::uint32_t & n_threads)
  {
	//total number of operations that must be performed by each thread
  	const std::uint32_t n_ops = n_elements / n_threads; 
	//may not divide evenly
	const std::uint32_t remaining_ops = n_elements % n_threads;
	//if it's the first thread, start should be 0
	const std::uint32_t start = (thread_id == 0) ? n_ops * thread_id : n_ops * thread_id + remaining_ops;
	const std::uint32_t stop = n_ops * (thread_id + 1) + remaining_ops;
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
  OutputIt paralell_copy_cpu(InputIt first, InputIt last, OutputIt output_first, const std::uint32_t & n_threads)
  {
	//to identify each thread
	std::uint32_t thread_id = 0;
	const std::uint32_t n_elements = std::distance(first, last);
	std::vector<std::thread> threads(n_threads);
	//initialize each thread
	for(std::thread & t : threads)
	{
	  t = std::thread(parallel_copy<InputIt,OutputIt>, std::ref(first), std::ref(output_first), thread_id, n_elements, n_threads );
	  ++thread_id;
	}
	for(std::thread & t : threads)
	  t.join();
	return output_first;
  }
  
  template< class InputIt, class T >
  T paralell_accumalute_cpu( InputIt first, InputIt last, T init,
									const std::uint32_t & n_threads = MAX_CPU_THREADS	  )
  {

	//to identify each thread
	std::uint32_t thread_id = 0;
	const std::uint32_t n_elements = std::distance(first, last);
	std::vector<std::thread> threads(n_threads);
	//initialize each thread
	for(std::thread & t : threads)
	{
	  t = std::thread(parallel_accumulate<InputIt,T>, std::ref(first), std::ref(init), thread_id, n_elements, n_threads );
	  ++thread_id;
	}
	for(std::thread & t : threads)
	{
	  t.join();
	}
	return init;
  }
  
  template < class InputIt, class UnaryFunction >
  UnaryFunction paralell_for_each_cpu(InputIt first, InputIt last, UnaryFunction f,
	  	                              const std::uint32_t & n_threads )
  
  {
	//to identify each thread
	std::uint32_t thread_id = 0;
	const std::uint32_t n_elements = std::distance(first, last);
	std::vector<std::thread> threads(n_threads);
	//initialize each thread
	for(std::thread & t : threads)
	{
	  t = std::thread(parallel_for_each<InputIt, UnaryFunction>, std::ref(first), std::ref(f), thread_id, n_elements, n_threads );
	  ++thread_id;
	}
	for(std::thread & t : threads)
	{
	  t.join();
	}
	return f;
  }
  template < class InputIt, class OutputIt, class UnaryOperation >
  OutputIt paralell_transform_cpu(InputIt first, InputIt last, OutputIt output_first, UnaryOperation unary_op, const std::uint32_t & n_threads )
  {
	//to identify each thread
	std::uint32_t thread_id = 0;
	const std::uint32_t n_elements = std::distance(first, last);
	std::vector<std::thread> threads(n_threads);
	//initialize each thread
	for(std::thread & t : threads)
	{
	  t = std::thread(parallel_transform<InputIt,OutputIt,UnaryOperation>, std::ref(first), std::ref(output_first), std::ref(unary_op), thread_id, n_elements, n_threads );
	  ++thread_id;
	}
	for(std::thread & t : threads)
	  t.join();
	return output_first;
  }
  template < class BidirectionalIt, class Generator >
  void paralell_generate_cpu(BidirectionalIt first, BidirectionalIt last, Generator g,
	   const std::uint32_t & n_threads)
  {
	//to identify each thread
	std::uint32_t thread_id = 0;
	const std::uint32_t n_elements = std::distance(first, last);
	std::vector<std::thread> threads(n_threads);
	//initialize each thread
	for(std::thread & t : threads)
	{
	  t = std::thread(parallel_generate<BidirectionalIt, Generator>, std::ref(first), std::ref(g), thread_id, n_elements, n_threads );
	  ++thread_id;
	}
	for(std::thread & t : threads)
	{
	  t.join();
	}
  }
}
