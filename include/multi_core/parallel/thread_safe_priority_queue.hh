#ifndef THREAD_SAFE_PRIORITY_QUEUE_HH
#define THREAD_SAFE_PRIORITY_QUEUE_HH
#include <multi_core/macros.hh>
#include <mutex>
#include <queue>
#include <condition_variable>
namespace zinhart
{
  namespace multi_core
  {
	template <class T, class Container = std::vector<T>, class Compare = std::less<typename Container::value_type>>
	  class thread_safe_priority_queue
	  {
		public:
		  HOST thread_safe_priority_queue();
		  // disable everthing that requires synchonization
		  HOST thread_safe_priority_queue(const thread_safe_priority_queue&) = delete;
		  HOST thread_safe_priority_queue(thread_safe_priority_queue&&) = delete;
		  HOST thread_safe_priority_queue & operator =(const thread_safe_priority_queue&) = delete;
		  HOST thread_safe_priority_queue & operator =(thread_safe_priority_queue&&) = delete;
		  HOST ~thread_safe_priority_queue();
		  const T & top();
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
		private:
		  enum class QUEUE_STATE : bool {ACTIVE = true, INACTIVE = false};
		  std::mutex lock;
		  std::priority_queue<T, Container, Compare> priority_queue;
		  std::condition_variable cv;
		  QUEUE_STATE queue_state;

	  };
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
#include <multi_core/parallel/ext/thread_safe_priority_queue.tcc>
#endif
