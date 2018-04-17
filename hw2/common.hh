#include <cstdint>
void saxpy(const std::uint32_t n_elements, const double a, double * x, double * y);
void launch_cpu_threaded_saxpy(
		const std::uint32_t n_elements, 
		const std::uint32_t n_threads,	const std::uint32_t n_threads,
		const double a, double * x, double * y
		);
void launch_gpu_threaded_saxpy(const std::uint32_t n_elements, double a, double * x, double * y);
