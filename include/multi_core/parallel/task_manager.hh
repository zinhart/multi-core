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
		  class task
		  {
			public:
			  task(std::uint32_t n_threads)
			  {
				this->n_threads = n_threads;
			  }
			  task(task && t)
			  {
				n_threads = t.threads();
				for(std::uint32_t i{0}; i < t.pending_futures.size(); ++i)
				  pending_tasks.push_back(t.pending_futures[i]);
			  }
			  task & operator = (task && t)
			  {
			  	n_threads = t.threads();
				for(std::uint32_t i{0}; i < t.pending_futures.size(); ++i)
				  pending_tasks.push_back(t.pending_futures[i]);
				return *this;
			  }
			  void add_future(thread_pool::tasks::task_future<T> && task_future)
			  { pending_futures.push_back(task_future); }

			  template<class Callable, class ... Args>
				HOST void push(std::uint64_t priority, Callable && c, Args&&...args)
				{
		  		  thread_pool::tasks::task_future<T> pending_future{thread_pool.add_task(priority, std::forward<Callable>(c), std::forward<Args>(args)...)};
		  		  pending_futures.push_back(std::move(pending_future));
				}
			  void wait()
			  {
				for(std::uint32_t i{0}; i < pending_futures.size(); ++i)
				  pending_futures[i].get();
			  }
			  void safe_wait()
			  {
				for(std::uint32_t i{0}; i < pending_futures.size(); ++i)
				  if(pending_futures[i].valid())
  					pending_futures[i].get();
			  }
			  std::uint32_t threads()
			  { return n_threads; }
			  ~task()
			  { safe_wait(); }
			private:
			  std::uint32_t n_threads;
			  std::vector< thread_pool::tasks::task_future<T> > pending_futures;
		  };
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
			HOST T push_wait(std::uint64_t priority, Callable && c, Args&&...args);
		  template<class Callable, class ... Args>
			HOST void push(std::uint64_t priority, Callable && c, Args&&...args);
		  HOST void push(task && t);
		  template<class Callable, class ... Args>
			HOST void push_at(std::uint64_t at, std::uint64_t priority, Callable && c, Args&&...args);

		private:
		  thread_pool::priority_pool thread_pool;
		  std::vector< thread_pool::tasks::task_future<T> > pending_tasks;
		  std::vector<task> pending_tasks_new;


	  };
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
#include <multi_core/parallel/ext/task_manager.tcc>
#endif
