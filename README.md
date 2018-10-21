# Multi_Core
[![Build Status](https://travis-ci.com/zinhart/multi_core.svg?branch=testing)](https://travis-ci.com/zinhart/multi_core)
[![Coverage Status](https://coveralls.io/repos/github/zinhart/multi_core/badge.svg?branch=testing)](https://coveralls.io/github/zinhart/multi_core?branch=testing)
#A Library of Cpu and Gpu Multi-Core Routines.

On the cpu side provides synchronous and asynchronous methods with a native thread pool.
I wrote this library for a few different reasons.

1. To provide myself a native means of speeding up my own implementations of machine learning algo's when I don't have access to a gpu.
2. At the time of this libraries creation isocpp parallelism's extensions to the stl are in the experimental namespace
3. To experiment with and teach myself multithreading in a c++ context
4. To centralize any and all parallel code I have in a single library which can be linked against

## CPU Build
 run ./build-cpu and choose a build option

## CPU Examples
  Using the thread pool:
  ```cpp
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
 ```
 A parallel std::generate
```cpp
 std::vector<zinhart::multi_core::thread_pool::task_future<void>> results;
 zinhart::multi_core::async::generate(x_parallel, x_parallel + n_elements, generator, results);
 for(i = 0; i < results.size(); ++i)
	results[i].get();
```
