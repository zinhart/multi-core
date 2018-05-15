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
	  class thread_task_interface
	  {
		  public:
		  	HOST thread_task_interface() = default;
		  	HOST thread_task_interface(const thread_task_interface&) = delete;
		  	HOST thread_task_interface & operator =(const thread_task_interface&) = delete;
		  	HOST thread_task_interface & operator =(thread_task_interface&&) = default;
		  	HOST ~thread_task_interface() = default;
		  	virtual void operator()() = 0;
	  };
	  template <class Callable>
		class thread_task : public thread_task_interface
		{
		  private:
			  Callable c;
		  public:
	  		  HOST thread_task(Callable && c)
		  	  {	this->c = std::move(c); }
		  	  HOST thread_task(const thread_task&) = delete;
		  	  HOST thread_task & operator =(const thread_task&) = delete;
		  	  HOST thread_task & operator =(thread_task&&) = default;
			  HOST virtual ~thread_task() = default;
			  void operator()() override
			  { this->c(); }
		};
	public:
	  template <class T>
		class thread_future
		{
		  private:
			  std::future<T> future;
		  public:
	  		  HOST thread_future(std::future<T> && future)
		  	  {	this->future = std::move(future); }
		  	  HOST thread_future(const thread_future&) = delete;
		  	  HOST thread_future & operator =(const thread_future&) = delete;
		  	  HOST thread_future & operator =(thread_future&&) = default;
			  HOST ~thread_future()
			  {
				if (future.valid())
			  		future.get();
			  }
			  T get()
			  {
				if (future.valid())
			  		return future.get();
			  }
		};
	public:
	  HOST void down();
	  // disable everthing
	  HOST thread_pool(const thread_pool&) = delete;
	  HOST thread_pool(thread_pool&&) = delete;
	  HOST thread_pool & operator =(const thread_pool&) = delete;
	  HOST thread_pool & operator =(thread_pool&&) = delete;
	  HOST thread_pool(std::uint32_t n_threads = std::max(1U, MAX_CPU_THREADS - 1));
	  HOST ~thread_pool(); 
	  HOST std::uint32_t get_threads();
	  
	  template<class Callable, class ... Args>
		HOST auto add_task(Callable && c, Args&&...args)
		{
		  auto bound_task = std::bind(std::forward<Callable>(c), std::forward<Args>(args)...);
		  using result_type = std::result_of_t<decltype(bound_task)()>;
		  using packaged_task = std::packaged_task<result_type()>;
		  using task_type = thread_task<packaged_task>;

		  packaged_task task{std::move(bound_task)};
		  thread_future<result_type> result{task.get_future()};
		  queue.push(std::make_unique<task_type>(std::move(task)));
		  return result;
		}
	private:
	  THREAD_POOL_STATE thread_pool_state;
	  std::vector<std::thread> threads;
	  thread_safe_queue< std::unique_ptr<thread_task_interface> > queue;
	  HOST void up(const std::uint32_t & n_threads);
	  HOST void work();
	};
	
}
#endif
