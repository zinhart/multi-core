#ifndef THREAD_POOL_TCC
#define THREAD_POOL_TCC
#include "concurrent_routines/thread_pool.hh"
#include <type_traits>
#include <memory>
namespace zinhart
{
  HOST void thread_pool::up(const std::uint32_t & n_threads)
  {
	for(std::uint32_t i = 0; i < n_threads; ++i )
	  threads.emplace_back(std::thread(&thread_pool::work, this));
	// set the queue state for work
	thread_pool_state = THREAD_POOL_STATE::UP;
  }	
  HOST void thread_pool::work()
  {
	while(thread_pool_state != THREAD_POOL_STATE::DOWN)
	{
	  std::function<void()> callable;
	  if(queue.pop(callable))
		callable();
	}
  }
  HOST void thread_pool::down()
  {
	thread_pool_state = THREAD_POOL_STATE::DOWN;
	for(std::thread & t : threads)
	  if(t.joinable())
		t.join();
  }
  HOST thread_pool::thread_pool(std::uint32_t n_threads)
  {
	  up(n_threads);
	  std::cout<<"Before work\n";
	  work();
	  std::cout<<"After work\n";
  }
  HOST thread_pool::~thread_pool()
  {
	std::cout<<"In destructor\n";
	down();
  }
  
  HOST std::uint32_t thread_pool::get_threads()
  {
	return threads.size();
  }

/*  template<class Callable, class ... Args>
	HOST auto thread_pool::add_task(Callable && c, Args&&...args) -> std::future<decltype(c(args...))>
	{
	  // wrap the given callable type and it's args with a function taking zero args  
	  std::function< decltype(c(args...)) ()> callable_task = std::bind(std::forward<Callable>(c), std::forward<Args>(args)...);
	  // store the callable in a shared pointer
	  std::shared_ptr< std::packaged_task< decltype( c(args...) )()>  > callable_task_ptr = std::make_shared<std::packaged_task< decltype( c(args...) )()> >(callable_task);
	  // now wrap the callable_task_ptr in a void function
	  std::function<void()> callable = [callable_task_ptr](){callable_task_ptr.get()();}; 
	  queue.push(callable);
	  return callable_task_ptr.get_future();
	}
*/
}
#endif
