#include <multi_core/multi_core.hh>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <limits>
using namespace testing;
//no exceptions segfaults
TEST(thread_pool, constructor_and_destructor)
{
  zinhart::multi_core::thread_pool::pool thread_pool;
}

TEST(thread_pool, call_add_task)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, MAX_CPU_THREADS);
  std::uniform_int_distribution<std::uint32_t> size_dist(1, MAX_CPU_THREADS);
  std::uint32_t results_size = size_dist(mt);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<std::uint32_t>> results;
  for(std::uint32_t i = 0, j = 0; i < results_size; ++i, ++j)
  {	  
	results.push_back(zinhart::multi_core::thread_pool::push_task([](std::uint32_t a, std::uint32_t b){ return a + b;}, i , j));
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
  zinhart::multi_core::thread_pool::tasks::task_future<std::int32_t> result = zinhart::multi_core::thread_pool::push_task([&t](){return t.add_one(3);});
  ASSERT_EQ(result.get(), 4);
}
std::int32_t plus_one(std::int32_t s)
{
  return s + 1;
}
TEST(thread_pool, call_add_task_function)
{
  zinhart::multi_core::thread_pool::tasks::task_future<std::int32_t> result = zinhart::multi_core::thread_pool::push_task(plus_one,3);
  ASSERT_EQ(result.get(), 4);
}

TEST(thread_pool, resize)
{	
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, 50);
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<std::uint32_t>> results;
  std::uint32_t i{0}, j{0};

  for(i = 0, j = 0; i < zinhart::multi_core::thread_pool::size(); ++i, ++j)
	results.push_back(zinhart::multi_core::thread_pool::push_task([](std::uint32_t a, std::uint32_t b){ return a + b;}, i , j));

  for(i = 0, j = 0; i < zinhart::multi_core::thread_pool::size(); ++i, ++j)
	ASSERT_EQ (i + j, results[i].get());

  const std::uint32_t new_pool_size = thread_dist(mt);
  zinhart::multi_core::thread_pool::resize(new_pool_size);
  ASSERT_EQ(zinhart::multi_core::thread_pool::size(), new_pool_size);

  results.clear();

  for(i = 0, j = 0; i <zinhart::multi_core::thread_pool::size(); ++i, ++j)
	results.push_back(zinhart::multi_core::thread_pool::push_task([](std::uint32_t a, std::uint32_t b){ return a + b;}, i , j));

  for(i = 0, j = 0; i < zinhart::multi_core::thread_pool::size(); ++i, ++j)
	ASSERT_EQ (i + j, results[i].get());

  ASSERT_EQ(i, new_pool_size); // since it should have iterated i times
}
