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
			public:
				//disable everthing
				HOST thread_safe_queue() = delete;
				HOST thread_safe_queue(const thread_safe_queue&) = delete;
				HOST thread_safe_queue(thread_safe_queue&&) = delete;
				HOST thread_safe_queue & operator =(const thread_safe_queue&) = delete;
				HOST thread_safe_queue & operator =(thread_safe_queue&&) = delete;
					void push(const T & item);
					void pop(T & item);
					void size();
					bool empty();

		};
}
#endif
