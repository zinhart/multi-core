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
#include <functional>
namespace zinhart
{
  //an asynchonous thread pool
  class thread_pool
  {
	enum class THREAD_POOL_STATE : bool {UP = true, DOWN = false};
	private:
	  THREAD_POOL_STATE thread_pool_state;
	  std::vector<std::thread> threads;
	  thread_safe_queue< std::function<void()> > queue;
	  HOST void up(const std::uint32_t & n_threads);
	  HOST void work();
	public:
	  HOST void down();
	  // disable everthing
	  //HOST thread_pool() = delete;
	  HOST thread_pool(const thread_pool&) = delete;
	  HOST thread_pool(thread_pool&&) = delete;
	  HOST thread_pool & operator =(const thread_pool&) = delete;
	  HOST thread_pool & operator =(thread_pool&&) = delete;
	  HOST ~thread_pool(); 
	  HOST thread_pool(std::uint32_t n_threads = std::max(1U, MAX_CPU_THREADS - 1));
	  HOST std::uint32_t get_threads();
	 /// template<class Callable, class ... Args>
		//HOST auto add_task(Callable && c, Args&&...args) -> std::future<decltype(c(args...))>;
	  template<class Callable, class ... Args>
		HOST auto add_task(Callable && c, Args&&...args) -> std::future<decltype(c(args...))>
		{
		  // wrap the given callable type and it's args with a function taking zero args  
		  std::function< decltype(c(args...)) ()> callable_task = std::bind(std::forward<Callable>(c), std::forward<Args>(args)...);
		  // store the callable in a shared pointer
		  std::shared_ptr< std::packaged_task< decltype( c(args...) )()>  > callable_task_ptr = std::make_shared<std::packaged_task< decltype( c(args...) )()> >(callable_task);
		  // now wrap the callable_task_ptr in a void function
		  std::function<void()> callable = [callable_task_ptr](){ (*callable_task_ptr)(); }; 
		  queue.push(callable);
		  return callable_task_ptr->get_future();
		}
	};
	
}
#endif
