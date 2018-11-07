#ifndef TASK_MANAGER_HH
#define TASK_MANAGER_HH
#include <multi_core/parallel/thread_pool.hh>
#include <iostream>
namespace zinhart
{
  namespace multi_core
  {
	template<class T>
	  class task_manager
	  {
		public:
		  HOST task_manager(const task_manager&) = delete;
		  HOST task_manager(task_manager&&) = delete;
		  HOST task_manager & operator =(const task_manager&) = delete;
		  HOST task_manager & operator =(task_manager&&) = delete;
		  HOST task_manager(std::uint32_t n_threads = std::max(1U, MAX_CPU_THREADS - 1));
		  HOST ~task_manager(); 
		  HOST T get(std::uint64_t index);	
		  HOST bool valid(std::uint64_t index);
		  HOST void resize(std::uint64_t n_threads);
		  HOST std::uint64_t size()const;
		  template<class Callable, class ... Args>
			HOST auto push_wait(std::uint64_t priority, Callable && c, Args&&...args)-> typename std::result_of<Callable(Args...)>::type
			{
			  // return type of future
			  using task_type = typename std::result_of<Callable(Args...)>::type;
			  // use std::forward to send to thread_pool::add_task
			  thread_pool::tasks::task_future<task_type> pending_task{thread_pool.add_task(priority, std::forward<Callable>(c), std::forward<Args>(args)...)};
			  return pending_task.get();
			}
		  template<class Callable, class ... Args>
			HOST void push(std::uint64_t priority, Callable && c, Args&&...args);

		private:
		  std::vector< thread_pool::tasks::task_future<T> > pending_tasks;// should be a task future not task interface
		  thread_pool::priority_pool thread_pool;

	  };
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
#include <multi_core/parallel/ext/task_manager.tcc>
#endif
