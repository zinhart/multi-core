#include "../thread_safe_queue.hh"
namespace zinhart
{
  template<class T>
	HOST thread_safe_queue<T>::thread_safe_queue()
			:state(queue_state::ACTIVE)
	{}
  template<class T>
	HOST thread_safe_queue<T>::~thread_safe_queue()
	{
	  kill();
	}
  template<class T>
	HOST typename thread_safe_queue<T>::queue_state thread_safe_queue<T>::current_state()
	{return (state == queue_state::ACTIVE) ? queue_state::ACTIVE : queue_state::INACTIVE; }

  template<class T>
	HOST void thread_safe_queue<T>::kill()
	{
	  //As I understand it when their is no cv or the need of locking multiple mutexes a std::lock_guard suffices
	  std::lock_guard<std::mutex> local_lock(lock);
	  //so that wait pop can exit
	  state = queue_state::INACTIVE;
	  //notify threads of the updated queue state
	  cv.notify_all();
	}
  template<class T>
	HOST void thread_safe_queue<T>::push(const T & item)
	{
	  std::lock_guard<std::mutex> temp_lock(lock);
	  //add item to the queue
	  queue.push(item);
	  //notify a thread that an item is to be removed from the queue
	  cv.notify_one();
	}
  template<class T>
	HOST bool thread_safe_queue<T>::try_pop(T & item)
	{
	  std::lock_guard<std::mutex> local_lock(lock);
	  //don't call pop on an empty queue
	  if(queue.empty())
	  {
		return false;
	  }
	  //avoid copying
	  item = std::move(queue.front());
	  //update queue
	  queue.pop();
	  return true;
	}
  template<class T>
	HOST bool thread_safe_queue<T>::wait_pop(T & item)
	{
	  //Since the cv is locked upon -> std::unique_lock
  	  std::unique_lock<std::mutex> local_lock(lock);
	  //basically block the current thread until the destructor is called (queue goes out of scope) or an item is available 
	  cv.wait(lock, [this](){ return current_state() == queue_state::INACTIVE || queue.size() > 0 ;}   );
	  // so that all threads can exit
	  if(current_state() == queue_state::INACTIVE)
		return false;
	  item = std::move(queue.front());
	  //update queue
	  queue.pop();
	  return true;
	}
  template<class T>
	HOST std::uint32_t thread_safe_queue<T>::size()
	{
	  std::lock_guard<std::mutex> local_lock(lock);
	  return queue.size();
	}
  template<class T>
	HOST bool thread_safe_queue<T>::empty()const
	{
	  std::lock_guard<std::mutex> local_lock(lock);
	  return queue.empty();
	}
  template<class T>
	HOST void clear()
	{
	  std::lock_guard<std::mutex> local_lock(lock);
	  while(queue.size() > 0)
		queue.pop();
	}
}
