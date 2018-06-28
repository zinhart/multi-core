#include "concurrent_routines/serial/serial.hh"
namespace zinhart
{
  namespace serial
  {
	// for embarrisingly parralell problems
	HOST void map(const std::uint32_t thread_id, const std::uint32_t & n_threads, const std::uint32_t & n_elements, std::uint32_t & start, std::uint32_t & stop)
	{
	  // total number of operations that must be performed by each thread
	  const std::uint32_t n_ops = n_elements / n_threads; 
	  
	  // may not divide evenly
	  const std::uint32_t remaining_ops = n_elements % n_threads;
	  
	  // the first thread will handle remaining opssee stop
	  start = (thread_id == 0) ? n_ops * thread_id : n_ops * thread_id + remaining_ops;
	  
	  // the index of the next start essentially
	  stop = n_ops * (thread_id + 1) + remaining_ops;
	}
	// from cuda samples for reduce
	HOST std::uint32_t next_pow2(std::uint32_t x)
	{
	  --x;
	  x |= x >> 1;
	  x |= x >> 2;
	  x |= x >> 4;
	  x |= x >> 8;
	  x |= x >> 16;
	  return ++x;
	}
	HOST void gemm_wrapper(std::int32_t & m, std::int32_t & n, std::int32_t & k, std::int32_t & lda, std::int32_t & ldb, std::int32_t & ldc, const std::uint32_t LDA, const std::uint32_t SDA, const std::uint32_t LDB, const std::uint32_t SDB)
	{
	  m = SDB;
	  n = LDA;
	  k = LDB;
	  lda = m;
	  ldb = k;
	  ldc = m;
	}
	HOST void geam_wrapper(std::int32_t & m, std::int32_t & n, std::int32_t & k, std::int32_t & lda, std::int32_t & ldb, std::int32_t & ldc, const std::uint32_t LDA, const std::uint32_t SDA, const std::uint32_t LDB, const std::uint32_t SDB)
	{
	  m = SDB;
	  n = LDA;
	  k = LDB;
	  lda = m;
	  ldb = k;
	  ldc = m;
	}
	CUDA_CALLABLE_MEMBER std::uint32_t idx2c(std::int32_t i,std::int32_t j,std::int32_t ld)// for column major ordering, if A is MxN then ld is M
	{ return j * ld + i; }
	CUDA_CALLABLE_MEMBER std::uint32_t idx2r(std::int32_t i,std::int32_t j,std::int32_t ld)// for row major ordering, if A is MxN then ld is N
	{ return i * ld + j; }
  }
}
