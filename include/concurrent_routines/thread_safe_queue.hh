#ifndef THREAD_SAFE_QUEUE_HH
#define THREAD_SAFE_QUEUE_HH
#include <mutex>
#include <queue>
namespace zinhart
{
	template <class T>
		class thread_safe_queue
		{
			private:
				mutex lock;
				std::queue<T> queue;
			public:
				HOST thread_safe_queue() = default;
				//disable everthing that requires synchonization
				HOST thread_safe_queue(const thread_safe_queue&) = delete;
				HOST thread_safe_queue(thread_safe_queue&&) = delete;
				HOST thread_safe_queue & operator =(const thread_safe_queue&) = delete;
				HOST thread_safe_queue & operator =(thread_safe_queue&&) = delete;
				HOST ~thread_safe_queue() = default;
				HOST void push(const T & task);
				HOST bool void pop(T & task);
				HOST std::uint32_t size();
				HOST bool empty();

		};
}
#endif
