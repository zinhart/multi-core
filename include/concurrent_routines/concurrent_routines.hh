#ifndef CONCURRENT_ROUNTINES_HH
#define CONCURRENT_ROUNTINES_HH
#include "macros.hh"
#include <cstdint>
namespace zinhart
{
  //CPU WRAPPERS
  void launch_cpu_threaded_saxpy(
	  const std::uint32_t n_elements, const std::uint32_t n_threads,
	  const double a, double * x, double * y
	  );
#if CUDA_ENABLED == true
  //GPU WRAPPERS
  void launch_gpu_threaded_saxpy(const std::uint32_t n_elements, double a, double * x, double * y);
#endif
}
#endif
