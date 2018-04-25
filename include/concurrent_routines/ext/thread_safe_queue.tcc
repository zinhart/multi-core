#include "../thread_safe_queue.hh"
namespace zinhart
{
	template<class T>
		HOST thread_safe_queue::push(const T & task)
		{
			std::unique_lock<std::mutex>(lock) temp_lock;
			queue.push(task);
		}
	template<class T>
		HOST bool thread_safe_queue::pop(const T & task)
		{
			std::unique_lock<std::mutex>(lock) temp_lock;
			//if the queue has tasks in it
			if(!queue.empty())
			{
				task = std::move(queue.front());
				queue.pop();	
				return true;
			}
			return false;
		}
	template<class T>
		HOST std::uint32_t thread_safe_queue::size()
		{
			std::unique_lock<std::mutex>(lock) temp_lock;
			return queue.size();
		}
	template<class T>
		HOST std::uint32_t thread_safe_queue::empty()
		{
			std::unique_lock<std::mutex>(lock) temp_lock;
			return queue.size();
		}
}
