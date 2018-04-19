#ifndef CONCURRENT_ROUNTINES_HH
#define CONCURRENT_ROUNTINES_HH
#include "macros.hh"
#include <thread>
#include <cstdint>
#define MAX_CPU_THREADS std::thread::hardware_concurrency()
namespace zinhart
{
  //CPU WRAPPERS
  void launch_cpu_threaded_saxpy(
  		const double a, double * x, double * y,
	  	const std::uint32_t n_elements, const std::uint32_t n_threads = MAX_CPU_THREADS
	
	  );
#if CUDA_ENABLED == true
  //GPU WRAPPERS
  void launch_gpu_threaded_saxpy(const std::uint32_t n_elements, double a, double * x, double * y);
#endif
}
#endif
