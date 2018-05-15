#ifndef THREAD_POOL_TCC
#define THREAD_POOL_TCC
#include "concurrent_routines/thread_pool.hh"
#include <type_traits>
#include <memory>
namespace zinhart
{
  HOST void thread_pool::up(const std::uint32_t & n_threads)
  {
	try
	{
 	 // set the queue state for work
	 thread_pool_state = THREAD_POOL_STATE::UP;
	 for(std::uint32_t i = 0; i < n_threads; ++i )
	  threads.emplace_back(&thread_pool::work, this);
	}
	catch(...)
	{
		down();
		throw;
	}
  }	
  HOST void thread_pool::work()
  {
	while(thread_pool_state != THREAD_POOL_STATE::DOWN)
	{
	  std::unique_ptr<thread_task_interface> task;
      if(queue.pop_on_available(task))
		(*task)();	  
	}

  }
  HOST void thread_pool::down()
  {
	thread_pool_state = THREAD_POOL_STATE::DOWN;
	queue.shutdown();
	for(std::thread & t : threads)
	  if(t.joinable())
		t.join();
  }
  HOST thread_pool::thread_pool(std::uint32_t n_threads)
  {
	  up(n_threads);
  }
  HOST thread_pool::~thread_pool()
  {
	std::cout<<"In destructor\n";
	down();
  }
  
  HOST std::uint32_t thread_pool::get_threads()
  {	return threads.size(); }

/*
  template<class Callable, class ... Args>
	HOST auto thread_pool::add_task(Callable && c, Args&&...args)
	{
	  auto bound_task = std::bind(std::forward<Callable>(c), std::forward<Args>(args)...);
	  using result_type = std::result_of_t<decltype(bound_task)()>;
	  using packaged_task = std::packaged_task<result_type()>;
	  using task_type = thread_task<packaged_task>;

	  packaged_task task = std::move(bound_task);
	  task_future<result_type> result{task.get_future()};
	  queue.push(std::make_unique<task_type>(std::move(task)));
	  return result;
	}*/
}
#endif
