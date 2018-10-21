#include <multi_core/multi_core.hh>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <limits>
using namespace testing;
TEST(thread_safe_queue, call_size_on_empty_queue)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(std::uint32_t{1}, std::uint32_t{MAX_CPU_THREADS});
  std::uint32_t n_threads = thread_dist(mt), i;
  std::vector<std::thread> threads(n_threads); 
  auto call_size = [](zinhart::multi_core::thread_safe_queue<std::int32_t>  & init_queue)
  {
	ASSERT_EQ(std::uint32_t{0}, init_queue.size());
  };
  zinhart::multi_core::thread_safe_queue<std::int32_t> test_queue;
  //called from the main thread
  call_size(test_queue);
  ASSERT_EQ(std::uint32_t{0}, test_queue.size());
  //call size from a random number of threads not exceding MAX_CPU_THREADS
  for( i = 0; i < n_threads; ++i)
  {
	//since this is an empty queue every thread should return false;
	threads[i] = std::thread(call_size, std::ref(test_queue) );
  }
  for(std::thread & t : threads)
  {
	t.join();
  }
}
TEST(thread_safe_queue, call_empty_on_empty_queue)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint8_t> thread_dist(1, MAX_CPU_THREADS);
  std::uint8_t n_threads = thread_dist(mt), i;
  std::vector<std::thread> threads(n_threads); 
  auto call_empty = [](zinhart::multi_core::thread_safe_queue<std::int32_t>  & init_queue)
  {
	ASSERT_EQ(true, init_queue.empty());
  };
  zinhart::multi_core::thread_safe_queue<std::int32_t> test_queue;
  //called from the main thread
  call_empty(test_queue);
  ASSERT_EQ(true, test_queue.empty());
  //call empty from a random number of threads not exceding MAX_CPU_THREADS
  for( i = 0; i < n_threads; ++i)
  {
	//since this is an empty queue every thread should return false;
	threads[i] = std::thread(call_empty, std::ref(test_queue) );
  }
  for(std::thread & t : threads)
  {
	t.join();
  }
}
TEST(thread_safe_queue, call_clear_on_empty_queue)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint8_t> thread_dist(1, MAX_CPU_THREADS);
  std::uint8_t n_threads = thread_dist(mt), i;
  std::vector<std::thread> threads(n_threads); 
  auto call_clear = [](zinhart::multi_core::thread_safe_queue<std::int32_t>  & init_queue)
  {
	init_queue.clear();
	//since these were already tested
	ASSERT_EQ(bool{true}, init_queue.empty());
	ASSERT_EQ(std::uint32_t{0}, init_queue.size());
  };
  zinhart::multi_core::thread_safe_queue<std::int32_t> test_queue;
  //called from the main thread
  call_clear(test_queue);
  ASSERT_EQ(bool{true}, test_queue.empty());
  ASSERT_EQ(std::uint32_t{0}, test_queue.size());
  //call clear from a random number of threads not exceding MAX_CPU_THREADS
  for( i = 0; i < n_threads; ++i)
  {
	//since this is an empty queue every thread should return false;
	threads[i] = std::thread(call_clear, std::ref(test_queue) );
  }
  for(std::thread & t : threads)
  {
	t.join();
  }
}

//this also implicitly tests push and size on nonempty queues
TEST(thread_safe_queue, call_push)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, MAX_CPU_THREADS);
  std::uint32_t n_threads = thread_dist(mt), i;
  std::vector<std::thread> threads(n_threads); 
  //to be called from each thread
  auto call_push = [](zinhart::multi_core::thread_safe_queue<std::uint32_t>  & init_queue, std::uint32_t item)
  {
	std::uint32_t old_size = init_queue.size();
	init_queue.push(item);
	ASSERT_TRUE( init_queue.size() >=  old_size);
  };
  zinhart::multi_core::thread_safe_queue<std::uint32_t> test_queue;
  call_push(test_queue,0);
  ASSERT_EQ(bool{false}, test_queue.empty());
  ASSERT_EQ(std::uint32_t{1}, test_queue.size());
  //call push from a random number of threads not exceding MAX_CPU_THREADS
  for( i = 0; i < n_threads; ++i)
  {
	//since this is an empty queue every thread should return false;
	threads[i] = std::thread(call_push, std::ref(test_queue), i + 1 );
  }
  for(std::thread & t : threads)
  {
	t.join();
  }
  //should be the queue size
  ASSERT_EQ( std::uint32_t{n_threads + 1}, test_queue.size());
}


TEST(thread_safe_queue, call_size_on_non_empty_queue)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, MAX_CPU_THREADS);
  std::uint32_t n_threads = thread_dist(mt), i;
  std::vector<std::thread> threads(n_threads); 
  //to be called from each thread
  auto call_push = [](zinhart::multi_core::thread_safe_queue<std::uint32_t>  & init_queue, std::uint32_t item)
  {
	std::uint32_t old_size = init_queue.size();
	init_queue.push(item);
	ASSERT_TRUE( init_queue.size() >=  old_size);
  };
  zinhart::multi_core::thread_safe_queue<std::uint32_t> test_queue;
  call_push(test_queue, 0);
  ASSERT_EQ(std::uint32_t{1}, test_queue.size());
  //call push from a random number of threads not exceding MAX_CPU_THREADS
  for( i = 0; i < n_threads; ++i)
  {
	//since this is an empty queue every thread should return false;
	threads[i] = std::thread(call_push, std::ref(test_queue), i + 1 );
  }
  for(std::thread & t : threads)
  {
	t.join();
  }
  //should be the queue size
  ASSERT_EQ( n_threads + 1, test_queue.size());
}

TEST(thread_safe_queue, call_empty_on_non_empty_queue)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint8_t> thread_dist(1, MAX_CPU_THREADS);
  std::uint8_t n_threads = thread_dist(mt), i;
  std::vector<std::thread> threads(n_threads); 
  //to be called from each thread
  auto call_push = [](zinhart::multi_core::thread_safe_queue<std::int32_t>  & init_queue, std::int32_t item)
  {
	init_queue.push(item);
	ASSERT_EQ(bool{false}, init_queue.empty());
  };
  zinhart::multi_core::thread_safe_queue<std::int32_t> test_queue;
  call_push(test_queue, 0);
  ASSERT_EQ(bool{false}, test_queue.empty());
  //call push from a random number of threads not exceding MAX_CPU_THREADS
  for( i = 0; i < n_threads; ++i)
  {
	//since this is an empty queue every thread should return false;
	threads[i] = std::thread(call_push, std::ref(test_queue), i + 1 );
  }
  for(std::thread & t : threads)
  {
	t.join();
  }
  //should not be empty
  ASSERT_EQ( bool{false}, test_queue.empty());
}

TEST(thread_safe_queue, call_pop_on_empty_queue)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint8_t> thread_dist(1, MAX_CPU_THREADS);
  std::uint8_t n_threads = thread_dist(mt), i;
  std::vector<std::thread> threads(n_threads); 
  //to be called from each thread
  auto call_pop = [](zinhart::multi_core::thread_safe_queue<std::int32_t>  & init_queue, std::int32_t item)
  {
	bool pop_result = init_queue.pop(item);
	ASSERT_EQ(bool{false}, pop_result);
  };
  zinhart::multi_core::thread_safe_queue<std::int32_t> test_queue;
  call_pop(test_queue, 0);
  for( i = 0; i < n_threads; ++i)
  {
	//since this is an empty queue every thread should return false;
	threads[i] = std::thread(call_pop, std::ref(test_queue), i + 1 );
  }
  for(std::thread & t : threads)
  {
	t.join();
  }
}

TEST(thread_safe_queue, call_pop_on_non_empty_queue)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint8_t> thread_dist(1, MAX_CPU_THREADS);
  std::uint8_t n_threads = thread_dist(mt), i;
  std::vector<std::thread> threads(n_threads); 
  //to be called from each thread
  auto call_push = [](zinhart::multi_core::thread_safe_queue<std::int32_t>  & init_queue, std::int32_t item)
  {
	init_queue.push(item);
  };
  zinhart::multi_core::thread_safe_queue<std::int32_t> test_queue;
  //call push from a random number of threads not exceding MAX_CPU_THREADS
  for( i = 0; i < n_threads; ++i)
  {
	//since this is an empty queue every thread should return false;
	threads[i] = std::thread(call_push, std::ref(test_queue), i + 1 );
  }
  for(std::thread & t : threads)
  {
	t.join();
  }
  //should be the queue size
  ASSERT_EQ( n_threads, test_queue.size());

  //now the queue has n_threads  elements
  auto test_call_pop = [](zinhart::multi_core::thread_safe_queue<std::int32_t>  & init_queue, std::int32_t & item, bool expected_result)
  {
	bool pop_result = init_queue.pop(item);
	ASSERT_EQ(expected_result, pop_result);
  };
  std::int32_t ret_val;
  for( i = 0; i < n_threads; ++i)
  {
	//since this is an empty queue every thread should return false;
	threads[i] = std::thread(test_call_pop, std::ref(test_queue), std::ref(ret_val), bool(true) );
  }

  for(std::thread & t : threads)
  {
	t.join();
  }
  ASSERT_EQ(test_queue.size(), std::uint32_t{0});
  //on empty queues pop should return false
  bool f = false;
  test_call_pop(std::ref(test_queue), ret_val, f );
}

//calling it on an empty queue doesn't make sense after all
TEST(thread_safe_queue, call_pop_on_available_on_non_empty_queue)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint8_t> thread_dist(1, MAX_CPU_THREADS);
  std::uint16_t n_threads = thread_dist(mt), i;
  std::vector<std::thread> threads(n_threads); 
  std::vector<std::future<void>> futures(n_threads);
  std::vector< std::packaged_task<void(zinhart::multi_core::thread_safe_queue<std::int32_t>&, std::int32_t&, bool)> > tasks(n_threads);

  for(i = 0; i < n_threads; ++i)
  {
	// create tasks
	std::packaged_task<void(zinhart::multi_core::thread_safe_queue<std::int32_t>&, std::int32_t&, bool)> test_call_pop(
	[](zinhart::multi_core::thread_safe_queue<std::int32_t>  & init_queue, std::int32_t & item, bool expected_result)
	{
	  init_queue.push(1);
	  bool pop_result = init_queue.pop_on_available(item);
      ASSERT_EQ(expected_result, pop_result);
	});
	// assign futures
	futures[i] = test_call_pop.get_future();
	// move tasks to a more convenient place
	tasks[i] = std::move(test_call_pop);
  }
  
  std::int32_t ret_val;
  
  zinhart::multi_core::thread_safe_queue<std::int32_t> test_queue;
  //call pop_on_available from a random number of threads not exceding MAX_CPU_THREADS
  for( i = 0; i < n_threads; ++i)
  {
	// since this is an empty queue every thread should return false;
	 threads[i] = std::thread(std::move(tasks[i]), std::ref(test_queue), std::ref(ret_val), true);
	 futures[i].get();
  }
  //destructor of queue called here
  for(std::thread & t : threads)
  {
	t.join();
  }

}
//no exceptions segfaults
TEST(thread_pool, constructor_and_destructor)
{
	zinhart::multi_core::thread_pool pool;
}
TEST(thread_pool, call_add_task)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, MAX_CPU_THREADS);
  std::uniform_int_distribution<std::uint32_t> size_dist(1, MAX_CPU_THREADS);
  std::uint32_t results_size = size_dist(mt);
  std::vector<zinhart::multi_core::thread_pool::task_future<std::uint32_t>> results;
  for(std::uint32_t i = 0, j = 0; i < results_size; ++i, ++j)
  {	  
	results.push_back(zinhart::multi_core::default_thread_pool::push_task([](std::uint32_t a, std::uint32_t b){ return a + b;}, i , j));
  }
  std::uint32_t res;
  for(std::uint32_t i = 0, j = 0; i < results_size; ++i, ++j)
  {	  
	res = results[i].get();  
	ASSERT_EQ(i + j, res);
  }
}
TEST(thread_pool, call_add_task_member_function)
{
  class test
  {
	public:
	  std::int32_t add_one(std::int32_t s)
	  {
	    return s + 1;
	  }
  };
  test t;
  zinhart::multi_core::thread_pool::task_future<std::int32_t> result = zinhart::multi_core::default_thread_pool::push_task([&t](){return t.add_one(3);});
  ASSERT_EQ(result.get(), 4);
}
std::int32_t plus_one(std::int32_t s)
{
  return s + 1;
}
TEST(thread_pool, call_add_task_function)
{
  zinhart::multi_core::thread_pool::task_future<std::int32_t> result = zinhart::multi_core::default_thread_pool::push_task(plus_one,3);
  ASSERT_EQ(result.get(), 4);
}
