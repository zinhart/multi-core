#include "../thread_safe_queue.hh"
#include <iostream>
namespace zinhart
{
  template<class T>
	HOST void thread_safe_queue<T>::push(const T & item)
	{
	  std::lock_guard<std::mutex> local_lock(lock);
	  // add item to the queue
	  queue.push(item);
	  // notify a thread that an item is ready to be removed from the queue
	  cv.notify_one();
	}
  template<class T>
	HOST void thread_safe_queue<T>::push(T && item)
	{
	  std::lock_guard<std::mutex> local_lock(lock);
	  // add item to the queue
	  queue.push(std::move(item));
	  // notify a thread that an item is ready to be removed from the queue
	  cv.notify_one();
	}
  template<class T>
	HOST bool thread_safe_queue<T>::pop(T & item)
	{
	  std::lock_guard<std::mutex> local_lock(lock);
	  if(queue.size() > 0)
	  {
		// avoid copying
		item = std::move(queue.front());
		// update queue
		queue.pop();
		// successfull write
		return true;
	  }
	  // unsuccessfull write
	  return false;
	}
  template<class T>
	HOST bool thread_safe_queue<T>::pop_on_available(T & item)
	{
	  // Since the cv is locked upon -> std::unique_lock
  	  std::unique_lock<std::mutex> local_lock(lock);
	  // basically block the current thread until an item is available, 
	  // so calling this function before pushing items on to the queue is an error,
	  // further wait conditions can be added here 
	  cv.wait(local_lock, [this](){ return queue.size() > 0; });
	  // avoid copying
	  item = std::move(queue.front());
	  // update queue
	  queue.pop();
	  // successfull write
	  return true;
	}

  template<class T>
	HOST std::uint32_t thread_safe_queue<T>::size()
	{
	  std::lock_guard<std::mutex> local_lock(lock);
	  return queue.size();
	}
  template<class T>
	HOST bool thread_safe_queue<T>::empty()
	{
	  std::lock_guard<std::mutex> local_lock(lock);
	  return queue.empty();
	}
  template<class T>
	HOST void thread_safe_queue<T>::clear()
	{
	  std::lock_guard<std::mutex> local_lock(lock);
	  while(queue.size() > 0)
		queue.pop();
	}
}
