#ifndef THREAD_SAFE_QUEUE_HH
#define THREAD_SAFE_QUEUE_HH
#include "macros.hh"
#include <mutex>
#include <queue>
#include <condition_variable>
namespace zinhart
{
  template <class T>
	class thread_safe_queue
	{
	  enum class queue_state : bool {ACTIVE = true, INACTIVE = false};
	  private:
		queue_state state;
		std::mutex lock;
		std::queue<T> queue;
		std::condition_variable cv;
		queue_state current_state();
		void kill();
	  public:
		HOST thread_safe_queue();
		//disable everthing that requires synchonization
		HOST thread_safe_queue(const thread_safe_queue&) = delete;
		HOST thread_safe_queue(thread_safe_queue&&) = delete;
		HOST thread_safe_queue & operator =(const thread_safe_queue&) = delete;
		HOST thread_safe_queue & operator =(thread_safe_queue&&) = delete;
		HOST ~thread_safe_queue();
		HOST void push(const T & item);
		//item only contains the value popped from the queue if the queue is not empty
		HOST bool try_pop(T & item);
		HOST bool wait_pop(T & item);
		//i.e pending items
		HOST std::uint32_t size();
		HOST bool empty() const;

	};
}
#include "ext/thread_safe_queue.tcc"
#endif
