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
		queue_state current_state;
		std::mutex lock;
		std::queue<T> queue;
		std::condition_variable cv;
		//queue_state current_state();
		void kill();
	  public:
		HOST thread_safe_queue();
		// disable everthing that requires synchonization
		HOST thread_safe_queue(const thread_safe_queue&) = delete;
		HOST thread_safe_queue(thread_safe_queue&&) = delete;
		HOST thread_safe_queue & operator =(const thread_safe_queue&) = delete;
		HOST thread_safe_queue & operator =(thread_safe_queue&&) = delete;
		HOST ~thread_safe_queue();
		HOST void push(const T & item);
		// item only contains the value popped from the queue if the queue is not empty
		HOST bool pop(T & item);
		// on pops when an item is available
		HOST bool pop_on_available(T & item);
		// i.e pending items
		HOST std::uint32_t size();
		HOST bool empty();
		HOST void clear();

	};
}
#include "ext/thread_safe_queue.tcc"
#endif
