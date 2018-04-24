#ifndef THREAD_POOL_HH
#define THREAD_POOL_HH
#include "macros.hh"
#include "thread_safe_queue.hh"
#include <thread>
#include <cstdint>
#include <future>
#include <mutex>
#include <vector>
namespace zinhart
{
	//an asynchonous thread pool
	class thread_pool
	{
		private:
			std::vector<std::thread> workers;
			std::mutex lock;
			std::condition_variable conditional_lock;
			enum class thread_pool_state : bool {ACTIVE = true, INACTIVE = false};
			// 1(true) is an active queue 0(false is inactive)
			bool state;
		public:
			//disable everthing
			HOST thread_pool() = delete;
			HOST thread_pool(const thread_pool&) = delete;
			HOST thread_pool(thread_pool&&) = delete;
			HOST thread_pool & operator =(const thread_pool&) = delete;
			HOST thread_pool & operator =(thread_pool&&) = delete;
			HOST thread_pool(std::uint32_t n_threads);
			HOST void init();
			HOST void shutdown();
			HOST auto add_task();
			
	};
}
#endif
