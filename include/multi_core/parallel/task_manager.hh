#ifndef TASK_MANAGER_HH
#define TASK_MANAGER_HH
#include <multi_core/parallel/thread_pool.hh>
#include <iostream>
namespace zinhart
{
  namespace multi_core
  {
	class task_manager
	{
	  public:

		enum class task_type : std::uint8_t {deffered = 0, immediate};
		template<task_type, class return_type, class ... parameters>
		  class task{};
		class task_interface
		{
		  public:
			task_interface()=default;
			task_interface(const task_interface & t);
			task_interface(task_interface && t)=default;
			task_interface & operator = (const task_interface & t)=default;
			task_interface & operator = (task_interface && t)=default;
			HOST void wait()
			{
			  wait_impl();
			}
			HOST void safe_wait()
			{
			  safe_wait_impl();
			}
			HOST void clear()
			{
			  clear_impl();
			}
			virtual ~task_interface()
			{ }

			template<class return_type, class ... parameters>
			  return_type result( task<task_type::immediate, return_type, parameters...> * t = nullptr)
			  {
				t = dynamic_cast<task<task_type::immediate, return_type, parameters...>*>(this);
				return (t != nullptr) ? t->result(): throw std::bad_cast();
			  }
		  private:
			HOST virtual void wait_impl() = 0;
			HOST virtual void safe_wait_impl() = 0;
			HOST virtual void clear_impl() = 0;

		};
		//immediate does not save args deffered does
		template<class return_type, class ... parameters>
		  class task<task_type::immediate, return_type, parameters...> : public task_interface// nearly finished, need to find a way to get the return value using task_interface
		  {
			public:
			  task(){};
			  explicit task(std::function<return_type(parameters...)> func)
				:function(std::move(func)){}
			  explicit task(std::function<return_type(parameters...)> func, parameters &&... params)
			  {
				function = std::move(func);
				push_back(std::forward<parameters>(params)...);
			  }
			  template <typename... Args>
				void push_back(Args&&... args) 
				{
				  pending_futures.push_back(thread_pool.add_task(0, function, std::forward<Args>(args)...));
				}
			  return_type result()
			  {
				return return_val;
			  }
			~task()
			{ safe_wait_impl(); }
			private:
			  std::function<return_type(parameters...)> function;
			  return_type return_val;
			  std::tuple<parameters...> stored_parameters;
			  std::vector< thread_pool::tasks::task_future<return_type> > pending_futures;
			  HOST void add_future(thread_pool::tasks::task_future<return_type> && task_future)
			  { pending_futures.push_back(task_future); }
			  HOST void wait_impl() override
			  {
				for(std::uint32_t i{0}; i < pending_futures.size(); ++i)
				  return_val = pending_futures[i].get();
			  }
			  HOST void safe_wait_impl() override
			  {
				for(std::uint32_t i{0}; i < pending_futures.size(); ++i)
				  if(pending_futures.at(i).valid())
				  {
					return_val = pending_futures.at(i).get();
				  }				
			  }
			  HOST void clear_impl() override
			  {
				safe_wait();
				pending_futures.clear();
			  }
		  };
		template<class return_type, class ... parameters>
		  class task<task_type::deffered, return_type(parameters...)> : public task_interface
		  {
		  // 	stored_parameters = std::make_tuple(std::forward<Args>(args)...);

			private:
			  std::function<return_type(parameters...)> function;
			  return_type result;
			  std::tuple<parameters...> stored_parameters;
			  std::vector< thread_pool::tasks::task_future<return_type> > pending_futures;
			  HOST virtual void wait_impl() override
			  {
			  }
			  HOST virtual void safe_wait_impl() override
			  {
			  }
			  HOST virtual void clear_impl() override
			  {
			  }
		  };
		template<class ... parameters>
		  class task<task_type::immediate, void(parameters...)> : public task_interface
		  {
			private:
			  HOST virtual void wait_impl() override
			  {
			  }
			  HOST virtual void safe_wait_impl() override
			  {
			  }
			  HOST virtual void clear_impl() override
			  {
			  }
		  };
		template<class ... parameters>
		  class task<task_type::deffered, void(parameters...)> : public task_interface
		  {
			private:
			  HOST virtual void wait_impl() override
			  {
			  }
			  HOST virtual void safe_wait_impl() override
			  {
			  }
			  HOST virtual void clear_impl() override
			  {
			  }
		  };
		// just a wrapper over 1 or more futures
	/*	template<class T>
		  class task : public task_interface
		  {
			public:
			  task(){};
			  task(task && t)
			  {	pending_futures = std::move(t.pending_futures); }
			  task & operator = (task && t)
			  {
				pending_futures = std::move(t.pending_futures);
				return *this;
			  }
			
			  template<class Callable, class ... Args>
				HOST void push(std::uint64_t priority, Callable && c, Args&&...args)
				{
				  thread_pool::tasks::task_future<T> pending_future{thread_pool.add_task(priority, std::forward<Callable>(c), std::forward<Args>(args)...)};
				}
			  ~task()
			  { safe_wait(); }

			private:
			  HOST void wait_impl() override
			  {
				for(std::uint32_t i{0}; i < pending_futures.size(); ++i)
				  pending_futures[i].get();
			  }
			  HOST void safe_wait_impl() override
			  {
				for(std::uint32_t i{0}; i < pending_futures.size(); ++i)
				  if(pending_futures.at(i).valid())
					pending_futures.at(i).get();
			  }
			  HOST void clear_impl() override
			  {
				safe_wait();
				pending_futures.clear();
			  }
			  void add_future(thread_pool::tasks::task_future<T> && task_future)
			  { pending_futures.push_back(task_future); }

			  std::uint32_t n_threads;
			  std::vector< thread_pool::tasks::task_future<T> > pending_futures;
		  };
		  */
		HOST task_manager(const task_manager&) = delete;
		HOST task_manager(task_manager&&) = delete;
		HOST task_manager & operator =(const task_manager&) = delete;
		HOST task_manager & operator =(task_manager&&) = delete;
		HOST task_manager(std::uint32_t n_threads = std::max(1U, MAX_CPU_THREADS - 1));
		HOST ~task_manager(); 
	//	HOST T get(std::uint64_t index);	
		HOST bool valid(std::uint64_t index);
		HOST void resize(std::uint64_t n_threads);
		HOST std::uint64_t size()const;
		/*
		template<class Callable, class ... Args>
		  HOST T push_wait(std::uint64_t priority, Callable && c, Args&&...args);
		template<class Callable, class ... Args>
		  HOST void push(std::uint64_t priority, Callable && c, Args&&...args);

		template<class Callable, class ... Args>
		  HOST void push_at(std::uint64_t at, std::uint64_t priority, Callable && c, Args&&...args);
		  */
		HOST void push(std::unique_ptr<task_interface> && t);
		HOST void wait();
		HOST void clear();
	  private:
		static thread_pool::priority_pool thread_pool;
	//	std::vector< thread_pool::tasks::task_future<T> > pending_tasks;
		std::queue<std::unique_ptr<task_interface>> pending_tasks;// queue


	};
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
//#include <multi_core/parallel/ext/task_manager.tcc>
#endif
