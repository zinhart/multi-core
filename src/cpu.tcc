#include "concurrent_routines/concurrent_routines_cpu_ext.hh"
#include <algorithm>
namespace zinhart
{
  template<>
	class copy<EXCECUTION_POLICY::SEQUENTIAL>
	{
	  public:
  		copy() = default;
  		copy(const copy &) = default;
  		copy(copy &&) = default;
  		copy & operator = (const copy &) = default;
  		copy & operator = (copy &&) = default;
  		~copy();
  		void operator ()()
	  	{
		}
	};
  template<>
	class copy<EXCECUTION_POLICY::PARALLEL>
	{
	  public:
  		copy() = default;
  		copy(const copy &) = default;
  		copy(copy &&) = default;
  		copy & operator = (const copy &) = default;
  		copy & operator = (copy &&) = default;
  		~copy();
  		void operator ()()
	  	{
		}
	};

  template<>
	class saxpy<EXCECUTION_POLICY::SEQUENTIAL>
	{
	  public:
  		saxpy() = default;
  		saxpy(const saxpy &) = default;
  		saxpy(saxpy &&) = default;
  		saxpy & operator = (const saxpy &) = default;
  		saxpy & operator = (saxpy &&) = default;
  		~saxpy();
		void operator ()(const std::uint32_t n_elements, double a, const double * x, double * y)
	  	{
	  	  for(std::uint32_t i = 0; i < n_elements; ++i)
	  	  {
			y[i] = a * x[i] + y[i];
		  }
		}
	};
  template<>
	class saxpy<EXCECUTION_POLICY::PARALLEL>
	{
	  public:
  		saxpy() = default;
  		saxpy(const saxpy &) = default;
  		saxpy(saxpy &&) = default;
  		saxpy & operator = (const saxpy &) = default;
  		saxpy & operator = (saxpy &&) = default;
  		~saxpy();
		void operator ()(const std::uint32_t thread_id,
						 const std::uint32_t n_threads, const std::uint32_t n_elements, 
						 const double a, double * x, double * y 
						)
		{
		  //total number of operations that must be performed by each thread
		  const std::uint32_t n_ops = n_elements / n_threads; 
		  //may not divide evenly
		  const std::uint32_t remaining_ops = n_elements % n_threads;
		  //if it's the first thread, start should be 0
		  const std::uint32_t start = (thread_id == 0) ? n_ops * thread_id : n_ops * thread_id + remaining_ops;
		  const std::uint32_t stop = n_ops * (thread_id + 1) + remaining_ops;
		  //operate on y's elements from start to stop
		  for(std::uint32_t op = start; op < stop; ++op)
		  {
			y[op] = a * x[op] + y[op];
		  }
		}
	};
}
