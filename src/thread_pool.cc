#include "concurrent_routines/thread_pool.hh"
namespace zinhart
{
	HOST thread_pool::thread_pool(std::uint32_t n_threads)
		:workers(std::vector<std::thread>(n_threads)), state(thread_pool_state::ACTIVE)
	{
		init();
	}
	HOST void init()
	{
		for(std::uint32_t i = 0; i < threads.size(); ++i)
		{
			//initialize each thread
		}
	}	
};
