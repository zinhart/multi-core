#ifndef THREAD_POOL_TCC
#define THREAD_POOL_TCC
#include "concurrent_routines/parallel/thread_pool.hh"
#include <type_traits>
#include <memory>
namespace zinhart
{
  namespace multi_core
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
	  std::shared_ptr<thread_task_interface> task;
	  while(thread_pool_state != THREAD_POOL_STATE::DOWN)
		if(queue.pop_on_available(task))
		  (*task)();	  
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
	{ up(n_threads); }

	HOST thread_pool::~thread_pool()
	{	down(); }
	
	HOST std::uint32_t thread_pool::size() const
	{	return threads.size(); }
	
	namespace default_thread_pool
	{
	  thread_pool & get_default_thread_pool()
	  {
		static thread_pool default_thread_pool;
		return default_thread_pool;
	  }

	}// END NAMESPACE DEFAULT_THREAD_POOL
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
#endif
