<h1>Using the thread pool:</h1>

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
