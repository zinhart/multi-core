#ifndef CONCURRENT_ROUTINES_HH
#define CONCURRENT_ROUTINES_HH
#include "macros.hh"
#include "parallel/parallel.hh"
#include "serial/serial.hh"
#include "timer.hh"
#include <cstdint>
namespace zinhart
{
#if CUDA_ENABLED == 1
	// Device properties
	namespace cuda_device_properties
	{
	  HOST auto get_properties(std::uint32_t device_id = 0) -> cudaDeviceProp;
	  HOST void get_warp_size(std::uint32_t & warp_size, const std::uint32_t & device_id);
	  HOST void get_max_shared_memory(std::uint32_t & max_shared_memory_per_block, const std::uint32_t & device_id);
	  HOST void get_max_threads_per_block(std::uint32_t & max_threads_per_block, const std::uint32_t & device_id);
	  HOST void get_max_threads_dim(std::int32_t (& max_threads_dim)[3], const std::uint32_t & device_id);
	  HOST void get_max_grid_size(std::int32_t (& max_threads_dim)[3], const std::uint32_t & device_id);
	  HOST void get_max_threads_1d_kernel(std::uint64_t & max_threads, const std::uint32_t & device_id);
	  template <std::uint32_t Kernel_Dim>
		class max_threads;
	  template<>
		class max_threads<1>
		{
		  public:
			max_threads() = default;
			max_threads(const max_threads&) = default;
			max_threads(max_threads&&) = default;
			max_threads & operator = (const max_threads&) = default;
			max_threads & operator = (max_threads&&) = default;
			~max_threads() = default;
			HOST static void get_max(std::uint64_t & max_threads, const std::uint32_t & device_id)
			{
			  std::int32_t max_grid_dim[3];
			  std::int32_t max_threads_dim[3];
			  get_max_threads_dim(max_threads_dim, device_id);
			  get_max_grid_size(max_grid_dim, device_id);
			  max_threads = max_threads_dim[0] * max_grid_dim[0];
			}
		};
	  template<>
		class max_threads<2>
		{
		  public:
			max_threads() = default;
			max_threads(const max_threads&) = default;
			max_threads(max_threads&&) = default;
			max_threads & operator = (const max_threads&) = default;
			max_threads & operator = (max_threads&&) = default;
			~max_threads() = default;
			HOST static void get_max(std::uint64_t & max_threads, const std::uint32_t & device_id)
			{
			  std::int32_t max_grid_dim[3];
			  std::int32_t max_threads_dim[3];
			  get_max_threads_dim(max_threads_dim, device_id);
			  get_max_grid_size(max_grid_dim, device_id);
			  max_threads = (max_threads_dim[0] * max_grid_dim[0]) + (max_threads_dim[1] * max_grid_dim[1]);
			}
		};
	  template<>
		class max_threads<3>
		{
		  public:
			max_threads() = default;
			max_threads(const max_threads&) = default;
			max_threads(max_threads&&) = default;
			max_threads & operator = (const max_threads&) = default;
			max_threads & operator = (max_threads&&) = default;
			~max_threads() = default;
			HOST static void get_max(std::uint64_t & max_threads, const std::uint32_t & device_id)
			{
			  std::int32_t max_grid_dim[3];
			  std::int32_t max_threads_dim[3];
			  get_max_threads_dim(max_threads_dim, device_id);
			  get_max_grid_size(max_grid_dim, device_id);
			  max_threads = (max_threads_dim[0] * max_grid_dim[0]) + (max_threads_dim[1] * max_grid_dim[1]) + (max_threads_dim[2] * max_grid_dim[2]);
			}
		};
	}
	// GPU HELPERS
	namespace grid_space
	{
	  template<std::uint32_t Grid_Dim>
		class grid;
	  template<>
		class grid<1>
		{
		  public:
			//assuming you only got 1 gpu
			HOST static void get_launch_params(dim3 & num_blocks, dim3 & threads_per_block, const std::uint32_t & N, const std::uint32_t & device_id = 0)
			{
			  std::uint32_t warp_size{0}; 
			  cuda_device_properties::get_warp_size(warp_size, device_id);
			  threads_per_block.x = (N + warp_size - 1) / warp_size * warp_size; // need to understand this
			  threads_per_block.y = 1;
			  threads_per_block.z = 1;
			  if(threads_per_block.x > 4 * warp_size) // 4 * warp_size = 128
				threads_per_block.x = 4 * warp_size;
			  num_blocks.x = (N + threads_per_block.x - 1) / threads_per_block.x;// number of blocks
			  num_blocks.y = 1;
			  num_blocks.z = 1;
			}
		};
	  //to do
	  template<>
		class grid<2>
		{
		  public:
			HOST static void get_launch_params(const std::uint32_t & n_elements, std::uint32_t & threads_per_block, std::uint32_t & x, std::uint32_t & y, std::uint32_t & z, const std::uint32_t & device_id = 0)
			{
			}
		};
	  //to do
	  template<>
		class grid<3>
		{
		  public:
			HOST static void get_launch_params(const std::uint32_t & n_elements, std::uint32_t & threads_per_block, std::uint32_t & x, std::uint32_t & y, std::uint32_t & z, const std::uint32_t & device_id = 0)
			{
			}
		};

	  // This function helps pick a grid size give N and the hardward limitations of the device
	  // N should be the number of outputs that the kernel will must compute
	  // returns 0 when N fits within the hardward specs 1 other wise
  	  HOST bool get_launch_params(dim3 & num_blocks, dim3 & threads_per_block, std::uint32_t N, const std::uint32_t & device_id = 0);
	  template <class T>
		HOST bool get_launch_params(dim3 & num_blocks, dim3 & threads_per_block, std::uint32_t N, std::uint32_t & elements_per_thread, const std::uint32_t & device_id = 0, T type = T{})
		{
	  	  std::uint64_t max_outputs_1d_kernel{0};
		  std::uint64_t max_outputs_2d_kernel{0};
		  std::uint64_t max_outputs_3d_kernel{0};
  		  std::uint32_t warp_size{0}; 
		  cuda_device_properties::max_threads<1>::get_max(max_outputs_1d_kernel, device_id); 
		  cuda_device_properties::max_threads<2>::get_max(max_outputs_2d_kernel, device_id); 
		  cuda_device_properties::max_threads<3>::get_max(max_outputs_3d_kernel, device_id); 
		  cuda_device_properties::get_warp_size(warp_size, device_id);

		  if(N <= max_outputs_1d_kernel)
		  {
			threads_per_block.x = (N + warp_size - 1) / warp_size * warp_size; // normalize and scale in terms of warp_size
			threads_per_block.y = 1;
			threads_per_block.z = 1;
			if(threads_per_block.x > 8 * warp_size) // 4 * warp_size = 128  
			  threads_per_block.x = 4 * warp_size;
			elements_per_thread = threads_per_block.x / warp_size + threads_per_block.x;
			num_blocks.x = /*elements_per_thread;*/(N + threads_per_block.x - 1) / threads_per_block.x;// number of blocks 
			num_blocks.y = 1;
			num_blocks.z = 1;
			return false;
		  }
		  else if (N <= max_outputs_2d_kernel && N > max_outputs_1d_kernel)
		  {
			// to do
			return false;
		  }
		  else if(N <= max_outputs_3d_kernel && N > max_outputs_2d_kernel)
		  {
			// to do
			return false;
		  }
		  else
		  {
			return true;
		  }
		}
	}
  /***************************
   * GPU WRAPPERS ************
   * *************************
   */
	template <class Precision_Type>
	  HOST std::int32_t call_axps(const Precision_Type & a, Precision_Type * x, const Precision_Type & s, const std::uint32_t & N, const std::uint32_t & device_id = 0);

	template <class Precision_Type>
	  HOST std::int32_t call_axps_async(const Precision_Type & a, Precision_Type * x, const Precision_Type & s, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id = 0);

	template <class Precision_Type>
	  HOST std::int32_t reduce(const Precision_Type * in, Precision_Type * out, const std::uint32_t & N, const std::uint32_t & device_id = 0);
	
	template <class Precision_Type>
	  HOST std::int32_t reduce(const Precision_Type * in, Precision_Type * out, const std::uint32_t & N, const cudaStream_t & stream, const std::uint32_t & device_id = 0);



	
#endif
}
#endif
