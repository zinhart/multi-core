#ifndef TASK_MANAGER_TCC
#define TASK_MANAGER_TCC
namespace zinhart
{
  namespace multi_core
  {
	template <class T>
  	  HOST task_manager<T>::task_manager(std::uint32_t n_threads)
	  { thread_pool.resize(n_threads); }
	
	template <class T>
	  HOST task_manager<T>::~task_manager()
	  {
		for(std::uint32_t task_id = 0; task_id < pending_tasks.size(); ++task_id)
		  if(valid(task_id))
			get(task_id);
	  } 

	template <class T>
	  HOST T task_manager<T>::get(std::uint64_t index)
	  { return pending_tasks.at(index).get(); }

	template <class T>
	  HOST bool task_manager<T>::valid(std::uint64_t index)
	  { return pending_tasks.at(index).valid(); }

	template <class T>
	  HOST void task_manager<T>::resize(std::uint64_t n_threads)
	  { 
		thread_pool.resize(n_threads);
	  }
	template <class T>
	  HOST std::uint64_t task_manager<T>::size()const
	  { 
		return thread_pool.size();
	  }
  
	template <class T>
	  template<class Callable, class ... Args>
	  HOST void task_manager<T>::push(std::uint64_t priority, Callable && c, Args&&...args)
	  {
		thread_pool::tasks::task_future<T> pending_task{thread_pool.add_task(priority, std::forward<Callable>(c), std::forward<Args>(args)...)};
		// wrap the future in a void function object
		pending_tasks.push_back(std::move(pending_task));
	  }

	
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
#endif
