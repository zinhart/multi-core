#ifndef CONCURRENT_ROUNTINES_HH
#define CONCURRENT_ROUNTINES_HH
#include "macros.hh"
#include <cstdint>
//CPU ROUTINES
void saxpy(const std::uint32_t n_elements, const double a, double * x, double * y);
void launch_cpu_threaded_saxpy(
		const std::uint32_t n_elements, const std::uint32_t n_threads,
		const double a, double * x, double * y
		);
//GPU ROUTINES
#if CUDA_ENABLED == true
void launch_gpu_threaded_saxpy(const std::uint32_t n_elements, double a, double * x, double * y);
#endif

#endif
