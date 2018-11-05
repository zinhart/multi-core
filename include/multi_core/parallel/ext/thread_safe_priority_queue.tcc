namespace zinhart
{
  namespace multi_core
  {
	template<class T>
	  HOST thread_safe_priority_queue<T>::thread_safe_priority_queue()
  	  { wakeup(); }
	template<class T>
	  HOST void thread_safe_priority_queue<T>::wakeup()
	  { 
		std::lock_guard<std::mutex> local_lock(lock);
		queue_state = QUEUE_STATE::ACTIVE; 
		cv.notify_all();
	  }
	template<class T>
	  HOST void thread_safe_priority_queue<T>::shutdown()
	  {
		std::lock_guard<std::mutex> local_lock(lock);
		queue_state = QUEUE_STATE::INACTIVE;
		cv.notify_all();
	  }
	template<class T>
	  HOST thread_safe_priority_queue<T>::~thread_safe_priority_queue()
	  { shutdown(); }
	template<class T>
	  HOST void thread_safe_priority_queue<T>::push(const T & item)
	  {
		std::lock_guard<std::mutex> local_lock(lock);
		// add item to the queue
		priority_queue.push(item);
		// notify a thread that an item is ready to be removed from the queue
		cv.notify_one();
	  }
	template<class T>
	  HOST void thread_safe_priority_queue<T>::push(T && item)
	  {
		std::lock_guard<std::mutex> local_lock(lock);
		// add item to the queue
		priority_queue.push(std::move(item));
		// notify a thread that an item is ready to be removed from the queue
		cv.notify_one();
	  }
	template<class T>
	  HOST bool thread_safe_priority_queue<T>::pop(T & item)
	  {
		std::lock_guard<std::mutex> local_lock(lock);
		if(priority_queue.size() > 0)
		{
		  // avoid copying
		  item = std::move(priority_queue.front());
		  // update queue
		  priority_queue.pop();
		  // successfull write
		  return true;
		}
		// unsuccessfull write
		return false;
	  }
	template<class T>
	  HOST bool thread_safe_priority_queue<T>::pop_on_available(T & item)
	  {
		// Since the cv is locked upon -> std::unique_lock
		std::unique_lock<std::mutex> local_lock(lock);
		// basically block the current thread until an item is available, 
		// so calling this function before pushing items on to the queue is an error,
		// further wait conditions could be added here 
		cv.wait(local_lock, [this](){ return priority_queue.size() > 0 || queue_state == QUEUE_STATE::INACTIVE; });
		
		// if an early termination signal is received then return an unsuccessfull write
		if (queue_state == QUEUE_STATE::INACTIVE)
			return false;
		// avoid copying
		item = std::move(priority_queue.front());
		// update queue
		priority_queue.pop();
		// successfull write
		return true;
	  }

	template<class T>
	  HOST std::uint32_t thread_safe_priority_queue<T>::size()
	  {
		std::lock_guard<std::mutex> local_lock(lock);
		return priority_queue.size();
	  }
	template<class T>
	  HOST bool thread_safe_priority_queue<T>::empty()
	  {
		std::lock_guard<std::mutex> local_lock(lock);
		return priority_queue.empty();
	  }
	template<class T>
	  HOST void thread_safe_priority_queue<T>::clear()
	  {
		std::lock_guard<std::mutex> local_lock(lock);
		while(priority_queue.size() > 0)
		  priority_queue.pop();
		cv.notify_all();
	  }
  }// END NAMESPACE MULTI_CORE
}// END NAMESPACE ZINHART
