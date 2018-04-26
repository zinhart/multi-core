#ifndef THREAD_POOL_HH
#define THREAD_POOL_HH
#include <mutex>
#include <cstdint>
#include <thread>
#include <future>
#include <functional>
#include <vector>
#include "macros.hh"
#include "thread_safe_queue.hh"
namespace zinhart
{
	//an asynchonous thread pool
	class thread_pool
	{
		private:
			std::mutex lock;
			std::condition_variable conditional_lock;
			enum class thread_pool_state : bool {ACTIVE = true, INACTIVE = false};
			// to record state
			bool current_state;
			std::vector<std::thread> workers;
			thread_safe_queue<std::function<void()>> queue;
		public:
			//disable everthing
			HOST thread_pool() = delete;
			HOST thread_pool(const thread_pool&) = delete;
			HOST thread_pool(thread_pool&&) = delete;
			HOST thread_pool & operator =(const thread_pool&) = delete;
			HOST thread_pool & operator =(thread_pool&&) = delete;
			HOST ~thread_pool() = default;
			HOST thread_pool(std::uint32_t n_threads);
			HOST void init();
			HOST void shutdown();
			HOST auto add_task();
	};
}
#endif
