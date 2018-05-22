#ifndef CONCURRENT_ROUTINES_HH
#define CONCURRENT_ROUTINES_HH
#include "macros.hh"
#include "timer.hh"
#include <thread>
#include <cstdint>
namespace zinhart
{
  /***************************
   * CPU WRAPPERS ************
   * *************************
   */
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


	// HELPER FUNCTIONS
	// this function is used by each thread to determine what pieces of data it will operate on
	HOST void map(const std::uint32_t thread_id, const std::uint32_t & n_threads, const std::uint32_t & n_elements, std::uint32_t & start, std::uint32_t & stop);
	CUDA_CALLABLE_MEMBER std::uint32_t idx2c(std::int32_t i,std::int32_t j,std::int32_t ld);// for column major ordering, if A is MxN then ld is M
	CUDA_CALLABLE_MEMBER std::uint32_t idx2r(std::int32_t i,std::int32_t j,std::int32_t ld);// for row major ordering, if A is MxN then ld is N
	template<class Precision_Type>
	  HOST void serial_matrix_product(Precision_Type * A, Precision_Type * B, Precision_Type * C, std::uint32_t m, std::uint32_t n, std::uint32_t k);
	template<class Precision_Type>
	  HOST void print_matrix_row_major(Precision_Type * mat, std::uint32_t mat_rows, std::uint32_t mat_cols, std::string s);

#if CUDA_ENABLED == 1
  /***************************
   * GPU WRAPPERS ************
   * *************************
   */
	template <class Precision_Type>
	  void reduce(std::uint32_t size, std::uint32_t threads, std::uint32_t blocks, Precision_Type * out, Precision_Type * in);
	// assumed to be row major indices this generated the column indices
    HOST std::int32_t gemm_wrapper(std::int32_t & m, std::int32_t & n, std::int32_t & k, std::int32_t & lda, std::int32_t & ldb, std::int32_t & ldc, const std::uint32_t LDA, const std::uint32_t SDA, const std::uint32_t LDB, std::uint32_t SDB);

	// GPU HELPERS
	 
	template<std::uint32_t Grid_Dim>
	  class grid;
	template<>
	  class grid<1>
	  {
		public:
		  //assuming you only got 1 gpu
		  void operator()(const std::uint32_t & N, std::uint32_t & threads_per_block, std::uint32_t & x, std::uint32_t & y, std::uint32_t & z, const std::uint32_t & device_id = 0)
		  {
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, 0);
			dim3 block_launch;
			std::uint32_t warp_size = properties.warpSize;
			threads_per_block = (N + warp_size -1) / warp_size * warp_size;
			if(threads_per_block > 4 * warp_size)
			  threads_per_block = 4 * warp_size;
			x = (N + threads_per_block - 1) / threads_per_block;// number of blocks
			y = 1;
			z = 1;
		  }
	  };
	//to do
	template<>
	  class grid<2>
	  {
		public:
		  void operator()(const std::uint32_t & n_elements, std::uint32_t & threads_per_block, std::uint32_t & x, std::uint32_t & y, std::uint32_t & z, const std::uint32_t & device_id = 0)
		  {
		  }
	  };
	//to do
	template<>
	  class grid<3>
	  {
		public:
		  void operator()(const std::uint32_t & n_elements, std::uint32_t & threads_per_block, std::uint32_t & x, std::uint32_t & y, std::uint32_t & z, const std::uint32_t & device_id = 0)
		  {
		  }
	  };
#endif
}
#include "ext/concurrent_routines_ext.tcc"
#endif
