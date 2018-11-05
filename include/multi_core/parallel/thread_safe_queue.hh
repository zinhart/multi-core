#ifndef THREAD_SAFE_QUEUE_HH
#define THREAD_SAFE_QUEUE_HH
#include <multi_core/macros.hh>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
namespace zinhart
{
  namespace multi_core
  {
	template <class T>
	  class thread_safe_queue
	  {
		enum class QUEUE_STATE : bool {ACTIVE = true, INACTIVE = false};
		private:
		  std::mutex lock;
		  std::queue<T> queue;
		  std::condition_variable cv;
		  QUEUE_STATE queue_state;
		public:
		  HOST thread_safe_queue();
		  // disable everthing that requires synchonization
		  HOST thread_safe_queue(const thread_safe_queue&) = delete;
		  HOST thread_safe_queue(thread_safe_queue&&) = delete;
		  HOST thread_safe_queue & operator =(const thread_safe_queue&) = delete;
		  HOST thread_safe_queue & operator =(thread_safe_queue&&) = delete;
		  HOST ~thread_safe_queue();
		  HOST void push(const T & item);
		  HOST void push(T && item);
		  // item only contains the value popped from the queue if the queue is not empty
		  HOST bool pop(T & item);
		  // blocks until queue.size() > 0
		  HOST bool pop_on_available(T & item);
		  // i.e pending items
		  HOST std::uint32_t size();
		  HOST bool empty();
		  HOST void clear();
		  HOST void wakeup();
		  //manually shutdown the queue
		  HOST void shutdown();
	  };
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
#include "ext/thread_safe_queue.tcc"
#endif
