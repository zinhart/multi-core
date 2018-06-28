#ifndef ZINHART_SERIAL_HH
#define ZINHART_SERIAL_HH
#include "../macros.hh"
#include <string>
#include <cstdint>
namespace zinhart
{
  namespace serial
  {
	template<class Precision_Type>
	  HOST void print_matrix_row_major(Precision_Type * mat, std::uint32_t mat_rows, std::uint32_t mat_cols, std::string s);
	  // HELPER FUNCTIONS
	  // this function is used by each thread to determine what pieces of data it will operate on
	  HOST void map(const std::uint32_t thread_id, const std::uint32_t & n_threads, const std::uint32_t & n_elements, std::uint32_t & start, std::uint32_t & stop);
	  // for reduce
	  HOST std::uint32_t next_pow2(std::uint32_t x);
	  CUDA_CALLABLE_MEMBER std::uint32_t idx2c(std::int32_t i,std::int32_t j,std::int32_t ld);// for column major ordering, if A is MxN then ld is M
	  CUDA_CALLABLE_MEMBER std::uint32_t idx2r(std::int32_t i,std::int32_t j,std::int32_t ld);// for row major ordering, if A is MxN then ld is N
	  template<class Precision_Type>
		HOST void serial_matrix_product(const Precision_Type * A, const Precision_Type * B, Precision_Type * C, const std::uint32_t m, const std::uint32_t n, const std::uint32_t k);
	  template<class Precision_Type>
		HOST void cache_aware_serial_matrix_product(const Precision_Type * A, const Precision_Type * B, Precision_Type * C, const std::uint32_t m, const std::uint32_t n, const std::uint32_t k);

	// assumed to be row major indices this generates the column indices
    HOST void gemm_wrapper(std::int32_t & m, std::int32_t & n, std::int32_t & k, std::int32_t & lda, std::int32_t & ldb, std::int32_t & ldc, const std::uint32_t LDA, const std::uint32_t SDA, const std::uint32_t LDB, std::uint32_t SDB);
  }
}
#include "ext/serial.tcc"
#endif
