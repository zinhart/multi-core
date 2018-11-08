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
Using the task manager:

```cpp
  // when you need to represent more than one return type 
  // a structure that is a wrapper over multiple data_types represented as a union
  struct example
  {
	enum class tags{uchar = 0, uint, string} tag;
	public:
	  example()
	  {
	  }
	  example(const example & e)
	  {
		if(e.get_tag() == tags::uchar)
		  set_uchar(e.get_uchar());
		else if(e.get_tag() == tags::uint)
		  set_uint(e.get_uint());
		else if(e.get_tag() == tags::string)
		  set_string(e.get_string());
		set_tag(get_tag());
	  }
	  example(example && e)
	  {
		if(e.get_tag() == tags::uchar)
		  set_uchar(e.get_uchar());
		else if(e.get_tag() == tags::uint)
		  set_uint(e.get_uint());
		else if(e.get_tag() == tags::string)
		  set_string(e.get_string());
		set_tag(get_tag());
	  }
	  example & operator = (const example & e)
	  {
		if(e.get_tag() == tags::uchar)
		  set_uchar(e.get_uchar());
		else if(e.get_tag() == tags::uint)
		  set_uint(e.get_uint());
		else if(e.get_tag() == tags::string)
		  set_string(e.get_string());
		set_tag(get_tag());
		return *this;
	  }

	  example & operator = (example && e)
	  {
		if(e.get_tag() == tags::uchar)
		  set_uchar(e.get_uchar());
		else if(e.get_tag() == tags::uint)
		  set_uint(e.get_uint());
		else if(e.get_tag() == tags::string)
		  set_string(e.get_string());
		set_tag(get_tag());
		return *this;
	  }
	  void set_uchar(std::uint8_t x)
	  {
		set_tag(tags::uchar);
		type.ui8 = x;
	  }

	  void set_uint(std::uint32_t x)
	  {
		set_tag(tags::uint);
		type.ui32 = x;
	  }
	  void set_string(std::string s)
	  {
		set_tag(tags::string);
		new (&type.str) std::string(s);
	  }
		std::uint8_t get_uchar()const
		{ return type.ui8; }
		std::uint32_t get_uint()const
		{ return type.ui32; }
		std::string get_string()const
		{ return type.str; }
		tags get_tag()const
		{ return tag; }
		void set_tag(tags tag)
		{ this->tag = tag; }
	private:
		union types
		{
		  types()
		  {}
		  types(const types & t)
		  {}
		  types(types && t)
		  {}
		  std::uint8_t ui8;
		  std::uint32_t ui32;
		  std::string str;
		  ~types()
		  {}
		} type;
  };

  zinhart::multi_core::task_manager<example> t;
  t.push(0, [](char a)
	{
	  example x; 
	  x.set_uchar(a);
	  return x;
	}, 'a'
  );
  std::cout<<t.get(0).get_uchar()<<"\n";
  
  t.push(0, [](std::uint32_t num)
	{
	  example x; 
	  x.set_uint(num);
	  return x;
	},
	1
  );
  std::cout<<t.get(1).get_uint()<<"\n";

  t.push(0, [](std::string s)
	{
	  example x; 
	  x.set_string(s);
	  return x;
	}, "apples"
  );
 example p(t.get(2));
 std::cout<<p.get_string()<<"\n";
```


