#ifndef THREAD_POOL_TCC
#define THREAD_POOL_TCC
#include <type_traits>
#include <memory>
#include <cassert>
namespace zinhart
{
  namespace multi_core
  {
	namespace thread_pool
	{
	  HOST void thread_pool<thread_safe_queue<std::shared_ptr<tasks::thread_task_interface>>>::up(const std::uint32_t & n_threads)
	  {
		try
		{
		  queue.wakeup();
		  // set the queue state for work
		  thread_pool_state = THREAD_POOL_STATE::UP;
		  for(std::uint32_t i = 0; i < n_threads; ++i )
		   threads.emplace_back(&thread_pool<thread_safe_queue<std::shared_ptr<tasks::thread_task_interface>>>::work, this);
		}
		catch(...)
		{
			down();
			throw;
		}
	  }	

	  HOST void thread_pool<thread_safe_queue<std::shared_ptr<tasks::thread_task_interface>>>::work()
	  {
		std::shared_ptr<tasks::thread_task_interface> task;
		while(thread_pool_state != THREAD_POOL_STATE::DOWN)
		  if(queue.pop_on_available(task))
			(*task)();	  
	  }

	  HOST void thread_pool<thread_safe_queue<std::shared_ptr<tasks::thread_task_interface>>>::down()
	  {
		thread_pool_state = THREAD_POOL_STATE::DOWN;
		queue.shutdown();
		for(std::thread & t : threads)
		  if(t.joinable())
			t.join();
		threads.clear();// new
	  }

	  HOST void thread_pool<thread_safe_queue<std::shared_ptr<tasks::thread_task_interface>>>::resize(std::uint32_t n_threads)
	  { 
		assert(n_threads > 0);
		down();
		up(n_threads);
	  }

	  HOST thread_pool<thread_safe_queue<std::shared_ptr<tasks::thread_task_interface>>>::thread_pool(std::uint32_t n_threads)
	  { up(n_threads); }

	  HOST thread_pool<thread_safe_queue<std::shared_ptr<tasks::thread_task_interface>>>::~thread_pool()
	  { down(); }

	  HOST std::uint32_t thread_pool<thread_safe_queue<std::shared_ptr<tasks::thread_task_interface>>>::size() const
	  { return threads.size(); }
	}// END NAMESPACE THREAD_POOL
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
#endif
