#include <multi_core/multi_core.hh>
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <limits>
#include <memory>
#include <functional>
#include <future>
#include <algorithm>

TEST(cpu_test_parallel, saxpy)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(50, std::numeric_limits<std::uint16_t>::max());
  std::uniform_int_distribution<std::uint16_t> thread_dist(MAX_CPU_THREADS, 50);
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  float alpha = real_dist(mt);
  const std::uint32_t n_threads{thread_dist(mt)};
  const std::uint32_t n_elements{uint_dist(mt)};
  float * x_parallel = new float [n_elements];
  float * y_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  float * y_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i{0}, thread_id{0};
  for(i = 0; i < n_elements; ++i )
  {
		float first = real_dist(mt);
		float second = real_dist(mt);
		x_serial[i] = first;
		y_serial[i] = second;
		x_parallel[i] = first;
		y_parallel[i] = second;
  }

  for(thread_id = 0; thread_id < n_threads; ++ thread_id)
	results.push_back(zinhart::multi_core::thread_pool::push_task(zinhart::multi_core::async::saxpy<float>, alpha, x_parallel, y_parallel, n_elements, n_threads, thread_id));
  zinhart::multi_core::async::saxpy<float>(alpha, x_serial, y_serial, n_elements);

  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  for(i = 0; i < n_elements; ++i)
  {
		ASSERT_EQ(y_parallel[i], y_serial[i]);
  }
  delete [] x_parallel;
  delete [] y_parallel;
  delete [] x_serial;
  delete [] y_serial;
}

TEST(cpu_test_parallel, copy)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> thread_dist(MAX_CPU_THREADS, 50);
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(50, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  const std::uint32_t n_threads{thread_dist(mt)};
  const std::uint32_t n_elements{uint_dist(mt)};
  float * x_parallel = new float [n_elements];
  float * y_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  float * y_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i{0};
  std::uint32_t thread_id{0};
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial[i] = first;
	x_parallel[i] = first;
  }
  
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
	results.push_back(zinhart::multi_core::thread_pool::push_task(zinhart::multi_core::async::copy<float>, x_parallel, y_parallel, n_elements, n_threads, thread_id ));

  zinhart::multi_core::async::copy<float>(x_serial, y_serial, n_elements);
  // make sure all threads are done with their portion before comparing the final result
  for(thread_id = 0; thread_id < results.size(); ++thread_id)
  {
	results[thread_id].get();
  }
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel[i], y_serial[i]);
  }
  delete [] x_parallel;
  delete [] y_parallel;
  delete [] x_serial;
  delete [] y_serial;
}

TEST(cpu_test_parallel, copy_if)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint16_t> thread_dist(MAX_CPU_THREADS, 50);
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  const std::uint32_t n_threads{thread_dist(mt)};
  const std::uint32_t n_elements{uint_dist(mt)};
  float * x_parallel = new float [n_elements];
  float * y_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  float * y_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i{0};
  std::uint32_t thread_id{0};
  for(i = 0; i < n_elements; ++i )
  {
		float first = real_dist(mt);
		x_serial[i] = first;
		x_parallel[i] = first;
  }
  auto unary_predicate = [](const float & init){ return (init >= 1.0) ? true : false ;};
  for(thread_id = 0; thread_id < n_threads; ++thread_id)
	results.push_back(zinhart::multi_core::thread_pool::push_task(zinhart::multi_core::async::copy_if<float,decltype(unary_predicate)>, x_parallel, y_parallel, unary_predicate, n_elements, n_threads, thread_id ));

  zinhart::multi_core::async::copy_if<float,decltype(unary_predicate)>( x_serial, y_serial, unary_predicate, n_elements);
  // make sure all threads are done with their portion before comparing the final result
  for(thread_id = 0; thread_id < results.size(); ++thread_id)
  {
	results[thread_id].get();
  }
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel[i], y_serial[i]);
  }
  delete [] x_parallel;
  delete [] y_parallel;
  delete [] x_serial;
  delete [] y_serial;
}
/*
TEST(cpu_test_parallel, replace)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  float * x_parallel = new float [n_elements];
  float * y_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  float * y_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i = 0;
  float old_value = real_dist(mt);
  float new_value = real_dist(mt);
  //effectively force a replace on all elements
  for(i = 0; i < n_elements; ++i )
  {
	x_serial[i] = old_value;
	//save the old values to 
	y_serial[i] = x_serial[i];
	x_parallel[i] = old_value;
  }
  zinhart::multi_core::async::replace(x_parallel, x_parallel + n_elements, old_value, new_value, results);
  std::replace(x_serial, x_serial + n_elements, old_value, new_value);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the new value 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel[i], new_value);
  }
  delete [] x_parallel;
  delete [] y_parallel;
  delete [] x_serial;
  delete [] y_serial;
}

TEST(cpu_test_parallel, replace_if)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  float * x_parallel = new float [n_elements];
  float * y_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  float * y_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i = 0;
  float old_value = real_dist(mt);
  float new_value = real_dist(mt);
  auto unary_predicate = [](float & init){return (init == 0.0) ? true : false;};
  //effectively force a replace on all elements
  for(i = 0; i < n_elements; ++i )
  {
	x_serial[i] = old_value;
	//save the old values to 
	y_serial[i] = x_serial[i];
	x_parallel[i] = old_value;
  }
  zinhart::multi_core::async::replace_if(x_parallel, x_parallel + n_elements, unary_predicate, new_value, results);
  std::replace_if(x_serial, x_serial + n_elements, unary_predicate, new_value);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the new value 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel[i], x_serial[i]);
  }
  delete [] x_parallel;
  delete [] y_parallel;
  delete [] x_serial;
  delete [] y_serial;
}


TEST(cpu_test_parallel, replace_copy)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  float * x_parallel = new float [n_elements];
  float * y_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  float * y_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i = 0;
  float new_value = real_dist(mt);
  float old_value = real_dist(mt);
  for(i = 0; i < n_elements; ++i )
  {
	x_serial[i] = old_value;
	x_parallel[i] = old_value;
  }
  zinhart::multi_core::async::replace_copy(x_parallel, x_parallel + n_elements, y_parallel, old_value, new_value, results );
  std::replace_copy(x_serial, x_serial + n_elements, y_serial, old_value, new_value);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel[i], y_serial[i]);
  }
  delete [] x_parallel;
  delete [] y_parallel;
  delete [] x_serial;
  delete [] y_serial;
}

TEST(cpu_test_parallel, replace_copy_if)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  float * x_parallel = new float [n_elements];
  float * y_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  float * y_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i = 0;
  float old_value = real_dist(mt);
  float new_value = real_dist(mt);
  auto unary_predicate = [](float & init){return (init == 0.0) ? true : false;};
  //effectively force a replace on all elements
  for(i = 0; i < n_elements; ++i )
  {
	x_serial[i] = old_value;
	x_parallel[i] = old_value;
  }
  zinhart::multi_core::async::replace_copy_if(x_parallel, x_parallel + n_elements, y_parallel, unary_predicate, new_value, results);
  std::replace_copy_if(x_serial, x_serial + n_elements, y_serial, unary_predicate, new_value);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the new value 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel[i], y_serial[i]);
  }
  delete [] x_parallel;
  delete [] y_parallel;
  delete [] x_serial;
  delete [] y_serial;
}

TEST(cpu_test_parallel, inner_product_first_overload)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  float * x_parallel = new float [n_elements];
  float * y_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  float * y_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i = 0;
  float first, second, init = real_dist(mt), parallel_ret, serial_ret;
  for(i = 0; i < n_elements; ++i )
  {
	first = real_dist(mt);
	second = real_dist(mt);
	x_serial[i] = first;
	y_serial[i] = second;
	x_parallel[i] = first;
	y_parallel[i] = second;
  }
  zinhart::multi_core::async::inner_product(x_parallel, x_parallel + n_elements, y_parallel, init, results);
  serial_ret = std::inner_product(x_serial, x_serial + n_elements, y_serial, init);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  } 
  parallel_ret = init;
  ASSERT_EQ(parallel_ret, serial_ret); 
  delete [] x_parallel;
  delete [] y_parallel;
  delete [] x_serial;
  delete [] y_serial;
}

TEST(cpu_test_parallel, inner_product_second_overload)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  float * x_parallel = new float [n_elements];
  float * y_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  float * y_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i = 0;
  float first, second, init = real_dist(mt), parallel_ret, serial_ret;
  for(i = 0; i < n_elements; ++i )
  {
	first = real_dist(mt);
	second = real_dist(mt);
	x_serial[i] = first;
	y_serial[i] = second;
	x_parallel[i] = first;
	y_parallel[i] = second;
  }
  zinhart::multi_core::async::inner_product(x_parallel, x_parallel + n_elements, y_parallel, init, std::plus<float>(), std::equal_to<float>(), results);
  serial_ret = std::inner_product(x_serial, x_serial + n_elements, y_serial, init, std::plus<float>(), std::equal_to<float>());
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  parallel_ret = init;
  ASSERT_EQ(parallel_ret, serial_ret);
  delete [] x_parallel;
  delete [] y_parallel;
  delete [] x_serial;
  delete [] y_serial;
}

TEST(cpu_test_parallel, accumulate)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  float * x_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial[i] = first;
	x_parallel[i] = first;
  }
  //sum
  float p_sum{0.0};
  zinhart::multi_core::async::accumulate(x_parallel, x_parallel + n_elements, p_sum, results);
  float s_sum = std::accumulate(x_serial, x_serial + n_elements, 0.0);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the same values 
  ASSERT_EQ(p_sum, s_sum);
  delete [] x_serial;
  delete [] x_parallel;
}

TEST(cpu_test_parallel, for_each)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  float * x_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial[i] = first;
	x_parallel[i] = first;
  }
  auto unary = []( float & a )
						{
						  a = a * 2.0;
						};
  zinhart::multi_core::async::for_each(x_parallel, x_parallel + n_elements, unary, results);
  std::for_each(x_serial, x_serial + n_elements, unary);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel[i], x_serial[i]);
  }
  delete [] x_serial;
  delete [] x_parallel;
}

TEST(cpu_test_parallel, transform)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  float * x_parallel = new float [n_elements];
  float * y_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  float * y_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i = 0;
  auto unary = []( float & a )
						{
						  a = a * 2.0;
						  return a;
						};
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial[i] = first;
	x_parallel[i] = first;
  }
  zinhart::multi_core::async::transform(x_parallel, x_parallel + n_elements, y_parallel, unary, results );
  std::transform(x_serial, x_serial + n_elements, y_serial, unary);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel[i], y_serial[i]);
  }
  delete [] x_parallel;
  delete [] y_parallel;
  delete [] x_serial;
  delete [] y_serial;
}

TEST(cpu_test_parallel, generate)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  float * x_parallel = new float [n_elements];
  float * x_serial = new float [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial[i] = first;
	x_parallel[i] = first;
  }
  auto generator = [](){ return -2.0; };
  zinhart::multi_core::async::generate(x_parallel, x_parallel + n_elements, generator, results);
  std::generate(x_serial, x_serial + n_elements, generator);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel[i], x_serial[i]);
  }
  delete [] x_serial;
  delete [] x_parallel;
}
// may delete these functions so tests comments for now
/*
TEST(cpu_test_parallel, kahan_sum)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  n_elements = 13;
  double * x_parallel = new double [n_elements];
  double * x_serial = new double [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i;
  double parallel_sum{0.0}, serial_sum{0.0};
  for(i = 0; i < n_elements; ++i )
  {
	double first = real_dist(mt);
	x_serial[i] = first;
	x_parallel[i] = first;
  }
  zinhart::multi_core::async::kahan_sum(x_parallel, n_elements, parallel_sum, results);
  serial_sum = zinhart::multi_core::kahan_sum(x_serial, n_elements);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  ASSERT_DOUBLE_EQ(parallel_sum, serial_sum);
  delete [] x_serial;
  delete [] x_parallel;
}


TEST(cpu_test_parallel, neumaier_sum)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  double * x_parallel = new double [n_elements];
  double * x_serial = new double [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i;
  double parallel_sum{0.0}, serial_sum{0.0};
  for(i = 0; i < n_elements; ++i )
  {
	double first = real_dist(mt);
	x_serial[i] = first;
	x_parallel[i] = first;
  }
  zinhart::multi_core::async::neumaier_sum(x_parallel, n_elements, parallel_sum, results);
  serial_sum = zinhart::multi_core::neumaier_sum(x_serial, n_elements);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  ASSERT_DOUBLE_EQ(parallel_sum, serial_sum);
  delete [] x_serial;
  delete [] x_parallel;
}
*/
/*
TEST(cpu_test_parallel, kahan_sum_two)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  double * x_parallel = new double [n_elements];
  double * y_parallel = new double [n_elements];
  double * x_serial = new double [n_elements];
  double * y_serial = new double [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i;
  double parallel_sum{0.0}, serial_sum{0.0};
  for(i = 0; i < n_elements; ++i )
  {
	double first = real_dist(mt), second = real_dist(mt);
	x_serial[i] = first;
	y_serial[i] = second;
	x_parallel[i] = first;
	y_parallel[i] = second;
  }
  auto add_two_scalars = [](double x, double y){return x + y;};
  zinhart::multi_core::async::kahan_sum(x_parallel, y_parallel, n_elements, parallel_sum, add_two_scalars, results);
  serial_sum = zinhart::multi_core::kahan_sum(x_serial, y_parallel, n_elements, add_two_scalars);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  ASSERT_DOUBLE_EQ(parallel_sum, serial_sum);
  delete [] x_serial;
  delete [] x_parallel;
  delete [] y_serial;
  delete [] y_parallel;
}

TEST(cpu_test_parallel, neumair_sum_two)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  double * x_parallel = new double [n_elements];
  double * y_parallel = new double [n_elements];
  double * x_serial = new double [n_elements];
  double * y_serial = new double [n_elements];
  std::vector<zinhart::multi_core::thread_pool::tasks::task_future<void>> results;
  std::uint32_t i;
  double parallel_sum{0.0}, serial_sum{0.0};
  for(i = 0; i < n_elements; ++i )
  {
	double first = real_dist(mt), second = real_dist(mt);
	x_serial[i] = first;
	y_serial[i] = second;
	x_parallel[i] = first;
	y_parallel[i] = second;
  }
  auto add_two_scalars = [](double x, double y){return x + y;};
  zinhart::multi_core::async::neumaier_sum(x_parallel, y_parallel, n_elements, parallel_sum, add_two_scalars, results);
  serial_sum = zinhart::multi_core::neumaier_sum(x_serial, y_parallel, n_elements, add_two_scalars);
  // make sure all threads are done with their portion before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  ASSERT_DOUBLE_EQ(parallel_sum, serial_sum);
  delete [] x_serial;
  delete [] x_parallel;
  delete [] y_serial;
  delete [] y_parallel;
}
TEST(cpu_test, serial_matrix_multiply)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  std::uniform_int_distribution<std::uint16_t> size_dist(1, 10 );
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(-1.0, 1.0);

  // Matrix dimensions
  const std::uint32_t M{size_dist(mt)};
  const std::uint32_t N{size_dist(mt)};
  const std::uint32_t K{size_dist(mt)};

  // Array sizes
  std::uint32_t A_elements = M * K;
  std::uint32_t B_elements = N * K;
  std::uint32_t C_elements = M * N;

  // Matrices
  double * A_cache_aware = new double [A_elements];
  double * B_cache_aware = new double [B_elements];
  double * C_cache_aware = new double [C_elements];
  double * A_naive = new double [A_elements];
  double * B_naive = new double [B_elements];
  double * C_naive = new double [C_elements];

  // Misc variables: their usage is contextual
  std::uint32_t i = 0;
  
  // Initialize all matrices
  for(i = 0; i < A_elements; ++i)
  {
	A_cache_aware[i] = real_dist(mt);
	A_naive[i] = A_cache_aware[i];
  }
  for(i = 0; i < B_elements; ++i)
  {
	B_cache_aware[i] = real_dist(mt);
	B_naive[i] = B_cache_aware[i];
  }
  for(i = 0; i < C_elements; ++i)
  {
	C_cache_aware[i] = 0.0;
	C_naive[i] = C_cache_aware[i];
  }

  zinhart::multi_core::cache_aware_serial_matrix_product(A_cache_aware, B_cache_aware, C_cache_aware, M, N, K);
  zinhart::multi_core::serial_matrix_product(A_naive, B_naive, C_naive, M, N, K);
  
  for(i = 0; i < C_elements; ++i)
  {
	ASSERT_EQ(C_cache_aware[i], C_naive[i])<<"i: "<< i <<" "<<__FILE__<< " "<<__LINE__<<"\n"; 
  }
  delete [] A_naive;
  delete [] B_naive;
  delete [] C_naive;
  delete [] A_cache_aware;
  delete [] B_cache_aware;
  delete [] C_cache_aware;
}

TEST(mkl_test, gemm)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_real_distribution<float> real_dist(-0.5, 0.5);
  std::uniform_int_distribution<std::uint16_t> size_dist(1, 30);
  double * A{nullptr};
  double * B{nullptr};
  double * C{nullptr};
  double * A_mkl{nullptr};
  double * B_mkl{nullptr};
  double * C_mkl{nullptr};
  std::uint32_t M, N, K;
  M = size_dist(mt);
  N = size_dist(mt);
  K = size_dist(mt);
  std::uint32_t A_row{M}, A_col{K}, B_row{K}, B_col{N}, C_row{M}, C_col{N};
  std::uint32_t i, j;
  double alpha{1.0}, beta{0.0};
  
  A =  (double *)mkl_malloc( A_row * A_col * sizeof( double ), 64 );
  B = (double *)mkl_malloc( B_row * B_col * sizeof( double ), 64 );
  C = (double *)mkl_malloc( C_row * C_col * sizeof( double ), 64 );
  A_mkl =  (double *)mkl_malloc( A_row * A_col * sizeof( double ), 64 );
  B_mkl = (double *)mkl_malloc( B_row * B_col * sizeof( double ), 64 );
  C_mkl = (double *)mkl_malloc( C_row * C_col * sizeof( double ), 64 );
  
  for(i = 0; i < A_row; ++i)
	for(j = 0; j < A_col; ++j)
	{
  	  A[zinhart::multi_core::idx2r(i, j, A_col)] = real_dist(mt);
	  A_mkl[zinhart::multi_core::idx2r(i, j, A_col)] = A[zinhart::multi_core::idx2r(i, j, A_col)];
	}

  for(i = 0; i < B_row; ++i)
	for(j = 0; j < B_col; ++j)
	{
  	  B[zinhart::multi_core::idx2r(i, j, B_col)] = real_dist(mt);
	  B_mkl[zinhart::multi_core::idx2r(i, j, B_col)] = B[zinhart::multi_core::idx2r(i, j, B_col)];
	}

  for(i = 0; i < C_row; ++i)
	for(j = 0; j < C_col; ++j)
	{
  	  C[zinhart::multi_core::idx2r(i, j, C_col)] = 0.0;
	  C_mkl[zinhart::multi_core::idx2r(i, j, C_col)] = C[zinhart::multi_core::idx2r(i, j, C_col)];
	}


  zinhart::multi_core::cache_aware_serial_matrix_product(A, B, C, M, N, K);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A_mkl, K, B_mkl, N, beta, C_mkl, N);
  for(i = 0; i < C_row; ++i)
	for(j = 0; j < C_col; ++j)
	  ASSERT_DOUBLE_EQ(C[zinhart::multi_core::idx2r(i, j, C_col)], C_mkl[zinhart::multi_core::idx2r(i, j, C_col)]);

  mkl_free( A );
  mkl_free( B );
  mkl_free( C );
  mkl_free( A_mkl );
  mkl_free( B_mkl );
  mkl_free( C_mkl );
}*/
