#include <multi_core/parallel/task_manager.hh>
#include <memory>
namespace zinhart
{
  namespace multi_core
  {
	thread_pool::priority_pool task_manager::thread_pool;
	HOST task_manager::task_manager(std::uint32_t n_threads)
	{ thread_pool.resize(n_threads); }
	
	HOST task_manager::~task_manager()
	{
	  for(std::uint32_t i = 0; i < pending_tasks.size(); ++i)
	  {
	  	auto task = std::move(pending_tasks.front());
		if(task != nullptr)
		  task->safe_wait();
		pending_tasks.pop();
	  }
		
	

	} 
/*
	template <class T>
	  HOST T task_manager<T>::get(std::uint64_t index)
	  { return pending_tasks.at(index).get(); }

	template <class T>
	  HOST bool task_manager<T>::valid(std::uint64_t index)
	  { return pending_tasks.at(index).valid(); }
*/
	  HOST void task_manager::resize(std::uint64_t n_threads)
	  { 
		thread_pool.resize(n_threads);
	  }
	  HOST std::uint64_t task_manager::size()const
	  { 
		return thread_pool.size();
	  }

	 HOST void task_manager::push(std::unique_ptr<task_interface> && t)
	 { pending_tasks.push(std::move(t)); }
	HOST void task_manager::wait()
	{

	  for(std::uint32_t i = 0; i < pending_tasks.size(); ++i)
	  {
	  	auto task = std::move(pending_tasks.front());
		if(task != nullptr)
		  task->safe_wait();
		pending_tasks.pop();
	  }
	}
	HOST void task_manager::clear()
	{
	  for(std::uint32_t i = 0; i < pending_tasks.size(); ++i)
	  {
	  	auto task = std::move(pending_tasks.front());
		if(task != nullptr)
		  task->clear();
		pending_tasks.pop();
	  }
	}
/*
	template <class T>
	  template<class Callable, class ... Args>
	  HOST T task_manager<T>::push_wait(std::uint64_t priority, Callable && c, Args&&...args)
	  {
		thread_pool::tasks::task_future<T> pending_task{thread_pool.add_task(priority, std::forward<Callable>(c), std::forward<Args>(args)...)};
		return pending_task.get();
	  }
  
	template <class T>
	  template<class Callable, class ... Args>
	  HOST void task_manager<T>::push(std::uint64_t priorety, Callable && c, Args&&...args)
	  {
		thread_pool::tasks::task_future<T> pending_task{thread_pool.add_task(priority, std::forward<Callable>(c), std::forward<Args>(args)...)};
		pending_tasks.push_back(std::move(pending_task));
	  }
*/
	
/*
	template <class T>
	  template<class Callable, class ... Args>
	  HOST void task_manager<T>::push_at(std::uint64_t at, std::uint64_t priority, Callable && c, Args&&...args)
	  {
		thread_pool::tasks::task_future<T> pending_task{thread_pool.add_task(priority, std::forward<Callable>(c), std::forward<Args>(args)...)};
		pending_tasks.at(at) = std::move(pending_task);
	  }
*/

	
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
