#include "concurrent_routines/concurrent_routines.hh"
//#include "concurrent_routines/thread_safe_queue.hh"
//#include "concurrent_routines/thread_pool.hh"
#include "gtest/gtest.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <limits>
#include <memory>
#include <functional>
#include <future>

TEST(cpu_test, parallel_saxpy)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  float alpha = real_dist(mt);
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
		float first = real_dist(mt);
		float second = real_dist(mt);
		x_serial.get()[i] = first;
		y_serial.get()[i] = second;
		x_parallel.get()[i] = first;
		y_parallel.get()[i] = second;
  }
  auto serial_saxpy = [](
						const float & a, float * x, float * y,
						const std::uint32_t & n_elements )
						{
						  for(std::uint32_t i = 0; i < n_elements; ++i)
						  {
							y[i] = a * x[i] + y[i];
						  }
						};
  zinhart::parallel::async::parallel_saxpy(alpha, x_parallel.get(), y_parallel.get(), n_elements, results);
  serial_saxpy(alpha, x_serial.get(), y_serial.get(), n_elements);
  // make sure all threads are done before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  for(i = 0; i < n_elements; ++i)
  {
		ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
  std::cout<<"Hello From CPU Tests\n";
}

TEST(cpu_test, parallel_copy)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial.get()[i] = first;
	x_parallel.get()[i] = first;
  }
  zinhart::parallel::async::parallel_copy(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), results );
  std::copy(x_serial.get(), x_serial.get() + n_elements, y_serial.get());
  // make sure all threads are done before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
}
/*
TEST(cpu_test, parallel_copy_if)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
		float first = real_dist(mt);
		x_serial.get()[i] = first;
		x_parallel.get()[i] = first;
  }
  auto unary_predicate = [](float & init){ return (init >= 1.0) ? true : false ;};
  zinhart::parallel::async::parallel_copy_if(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), unary_predicate, results );
  std::copy_if(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), unary_predicate);
  // make sure all threads are done before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
}

TEST(cpu_test, parallel_replace)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  float old_value = real_dist(mt);
  float new_value = real_dist(mt);
  //effectively force a replace on all elements
  for(i = 0; i < n_elements; ++i )
  {
	x_serial.get()[i] = old_value;
	//save the old values to 
	y_serial.get()[i] = x_serial.get()[i];
	x_parallel.get()[i] = old_value;
  }
  zinhart::parallel::async::parallel_replace(x_parallel.get(), x_parallel.get() + n_elements, old_value, new_value, results);
  std::replace(x_serial.get(), x_serial.get() + n_elements, old_value, new_value);
  // make sure all threads are done before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the new value 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel.get()[i], new_value);
  }
}

TEST(cpu_test, parallel_replace_if)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
	float old_value = real_dist(mt);
	float new_value = real_dist(mt);
	auto unary_predicate = [](float & init){return (init == 0.0) ? true : false;};
	//effectively force a replace on all elements
  for(i = 0; i < n_elements; ++i )
  {
	x_serial.get()[i] = old_value;
	//save the old values to 
	y_serial.get()[i] = x_serial.get()[i];
	x_parallel.get()[i] = old_value;
  }
  zinhart::parallel::async::parallel_replace_if(x_parallel.get(), x_parallel.get() + n_elements, unary_predicate, new_value);
  std::replace_if(x_serial.get(), x_serial.get() + n_elements, unary_predicate, new_value);
  // make sure all threads are done before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the new value 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel.get()[i], x_serial.get()[i]);
  }
}


TEST(cpu_test, parallel_replace_copy)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  float new_value = real_dist(mt);
  float old_value = real_dist(mt);
  for(i = 0; i < n_elements; ++i )
  {
	x_serial.get()[i] = old_value;
	x_parallel.get()[i] = old_value;
  }
  zinhart::parallel::async::parallel_replace_copy(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), old_value, new_value );
  std::replace_copy(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), old_value, new_value);
  // make sure all threads are done before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
}


TEST(cpu_test, parallel_replace_copy_if)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  float old_value = real_dist(mt);
  float new_value = real_dist(mt);
  auto unary_predicate = [](float & init){return (init == 0.0) ? true : false;};
  //effectively force a replace on all elements
  for(i = 0; i < n_elements; ++i )
  {
	x_serial.get()[i] = old_value;
	x_parallel.get()[i] = old_value;
  }
  zinhart::parallel::async::parallel_replace_copy_if(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), unary_predicate, new_value);
  std::replace_copy_if(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), unary_predicate, new_value);
  // make sure all threads are done before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the new value 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
}
*/
/*
TEST(cpu_test, parallel_sample)
{


}
*/
/*
 * take special care with inner prod
TEST(cpu_test, parallel_inner_product_first_overload)
{
 	std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  float first, second, init = real_dist(mt), parallel_ret, serial_ret;
  for(i = 0; i < n_elements; ++i )
  {
	first = real_dist(mt);
	second = real_dist(mt);
	x_serial.get()[i] = first;
	y_serial.get()[i] = second;
	x_parallel.get()[i] = first;
	y_parallel.get()[i] = second;
  }
  parallel_ret =  zinhart::parallel::async::parallel_inner_product(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), init);
  serial_ret = std::inner_product(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), init);
  ASSERT_EQ(parallel_ret, serial_ret); 
}

TEST(cpu_test, parallel_inner_product_second_overload)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  float first, second, init = real_dist(mt), parallel_ret, serial_ret;
  for(i = 0; i < n_elements; ++i )
  {
	first = real_dist(mt);
	second = real_dist(mt);
	x_serial.get()[i] = first;
	y_serial.get()[i] = second;
	x_parallel.get()[i] = first;
	y_parallel.get()[i] = second;
  }
  parallel_ret =  zinhart::parallel::async::parallel_inner_product(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), init, std::plus<float>(), std::equal_to<float>());
  serial_ret = std::inner_product(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), init,
	  std::plus<float>(), std::equal_to<float>());
  ASSERT_EQ(parallel_ret, serial_ret);
}

TEST(cpu_test, parallel_accumulate)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial.get()[i] = first;
	x_parallel.get()[i] = first;
  }
  //sum
  float p_sum = zinhart::parallel::async::parallel_accumulate(x_parallel.get(), x_parallel.get() + n_elements, 0.0);
  float s_sum = std::accumulate(x_serial.get(), x_serial.get() + n_elements, 0);
  //double check we have the same values 
  ASSERT_EQ(p_sum,s_sum);
}

TEST(cpu_test, parallel_for_each)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial.get()[i] = first;
	x_parallel.get()[i] = first;
  }
  auto unary = []( float & a )
						{
						  a = a * 2.0;
						};
  zinhart::parallel::async::parallel_for_each(x_parallel.get(), x_parallel.get() + n_elements, unary);
  std::for_each(x_serial.get(), x_serial.get() + n_elements, unary);
  // make sure all threads are done before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel.get()[i], x_serial.get()[i]);
  }
}


TEST(cpu_test, parallel_transform)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> y_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  auto unary = []( float & a )
						{
						  a = a * 2.0;
						  return a;
						};
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial.get()[i] = first;
	x_parallel.get()[i] = first;
  }
  zinhart::parallel::async::parallel_transform(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(),unary );
  std::transform(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), unary);
  // make sure all threads are done before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
}

TEST(cpu_test, parallel_generate)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<float> x_parallel = std::shared_ptr<float>(new float [n_elements]);
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::vector<zinhart::parallel::thread_pool::task_future<void>> results;
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial.get()[i] = first;
	x_parallel.get()[i] = first;
  }
  auto generator = [](){ return -2.0; };
  zinhart::parallel::async::parallel_generate(x_parallel.get(), x_parallel.get() + n_elements, generator);
  std::generate(x_serial.get(), x_serial.get() + n_elements, generator);
  // make sure all threads are done before comparing the final result
  for(i = 0; i < results.size(); ++i)
  {
	results[i].get();
  }
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel.get()[i], x_serial.get()[i]);
  }
}
*/
/*TEST(cpu_test, serial_matrix_multiply)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  std::uniform_int_distribution<std::uint16_t> size_dist(1, std::numeric_limits<std::uint8_t>::max());
  //for any needed random real
  std::uniform_real_distribution<float> real_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());

  // Matrix dimensions
  const std::uint32_t M{size_dist(mt)};
  const std::uint32_t N{size_dist(mt)};
  const std::uint32_t K{size_dist(mt)};

  // Array sizes
  std::uint32_t A_elements = M * K;
  std::uint32_t B_elements = N * K;
  std::uint32_t C_elements = M * N;

  // Matrices
  std::uint32_t * A_cache_aware = new std::uint32_t [A_elements];
  std::uint32_t * B_cache_aware = new std::uint32_t [B_elements];
  std::uint32_t * C_cache_aware = new std::uint32_t [C_elements];
  std::uint32_t * A_naive = new std::uint32_t [A_elements];
  std::uint32_t * B_naive = new std::uint32_t [B_elements];
  std::uint32_t * C_naive = new std::uint32_t [C_elements];

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
	C_cache_aware[i] = real_dist(mt);
	C_naive[i] = C_cache_aware[i];
  }

  zinhart::serial::cache_aware_serial_matrix_product(A_cache_aware, B_cache_aware, C_cache_aware, M, N, K);
  zinhart::serial::serial_matrix_product(A_naive, B_naive, C_naive, M, N, K);
  
  for(i = 0; i < C_elements; ++i)
  {
	ASSERT_EQ(C_cache_aware[i], C_naive[i])<<"i: "<< i <<" "<<__FILE__<< " "<<__LINE__<<"\n"; 
  }

}*/
TEST(thread_safe_queue, call_size_on_empty_queue)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(std::uint32_t{1}, std::uint32_t{MAX_CPU_THREADS});
  std::uint32_t n_threads = thread_dist(mt), i;
  std::vector<std::thread> threads(n_threads); 
  auto call_size = [](zinhart::parallel::thread_safe_queue<std::int32_t>  & init_queue)
  {
	ASSERT_EQ(std::uint32_t{0}, init_queue.size());
  };
  zinhart::parallel::thread_safe_queue<std::int32_t> test_queue;
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
  auto call_empty = [](zinhart::parallel::thread_safe_queue<std::int32_t>  & init_queue)
  {
	ASSERT_EQ(true, init_queue.empty());
  };
  zinhart::parallel::thread_safe_queue<std::int32_t> test_queue;
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
  auto call_clear = [](zinhart::parallel::thread_safe_queue<std::int32_t>  & init_queue)
  {
	init_queue.clear();
	//since these were already tested
	ASSERT_EQ(bool{true}, init_queue.empty());
	ASSERT_EQ(std::uint32_t{0}, init_queue.size());
  };
  zinhart::parallel::thread_safe_queue<std::int32_t> test_queue;
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
  auto call_push = [](zinhart::parallel::thread_safe_queue<std::uint32_t>  & init_queue, std::uint32_t item)
  {
	std::uint32_t old_size = init_queue.size();
	init_queue.push(item);
	ASSERT_TRUE( init_queue.size() >=  old_size);
  };
  zinhart::parallel::thread_safe_queue<std::uint32_t> test_queue;
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
  auto call_push = [](zinhart::parallel::thread_safe_queue<std::uint32_t>  & init_queue, std::uint32_t item)
  {
	std::uint32_t old_size = init_queue.size();
	init_queue.push(item);
	ASSERT_TRUE( init_queue.size() >=  old_size);
  };
  zinhart::parallel::thread_safe_queue<std::uint32_t> test_queue;
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
  auto call_push = [](zinhart::parallel::thread_safe_queue<std::int32_t>  & init_queue, std::int32_t item)
  {
	init_queue.push(item);
	ASSERT_EQ(bool{false} ,init_queue.empty());
  };
  zinhart::parallel::thread_safe_queue<std::int32_t> test_queue;
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
  auto call_pop = [](zinhart::parallel::thread_safe_queue<std::int32_t>  & init_queue, std::int32_t item)
  {
	bool pop_result = init_queue.pop(item);
	ASSERT_EQ(bool{false}, pop_result);
  };
  zinhart::parallel::thread_safe_queue<std::int32_t> test_queue;
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
  auto call_push = [](zinhart::parallel::thread_safe_queue<std::int32_t>  & init_queue, std::int32_t item)
  {
	init_queue.push(item);
  };
  zinhart::parallel::thread_safe_queue<std::int32_t> test_queue;
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
  auto test_call_pop = [](zinhart::parallel::thread_safe_queue<std::int32_t>  & init_queue, std::int32_t & item, bool expected_result)
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
  std::vector< std::packaged_task<void(zinhart::parallel::thread_safe_queue<std::int32_t>&, std::int32_t&, bool)> > tasks(n_threads);

  for(i = 0; i < n_threads; ++i)
  {
	// create tasks
	std::packaged_task<void(zinhart::parallel::thread_safe_queue<std::int32_t>&, std::int32_t&, bool)> test_call_pop(
	[](zinhart::parallel::thread_safe_queue<std::int32_t>  & init_queue, std::int32_t & item, bool expected_result)
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
  
  zinhart::parallel::thread_safe_queue<std::int32_t> test_queue;
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
	zinhart::parallel::thread_pool pool;
}
TEST(thread_pool, call_add_task)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::uint32_t> thread_dist(1, MAX_CPU_THREADS);
  std::uniform_int_distribution<std::uint32_t> size_dist(1, MAX_CPU_THREADS);
  std::uint32_t results_size = size_dist(mt);
  std::vector<zinhart::parallel::thread_pool::task_future<std::uint32_t>> results;
  for(std::uint32_t i = 0, j = 0; i < results_size; ++i, ++j)
  {	  
	results.push_back(zinhart::parallel::default_thread_pool::push_task([](std::uint32_t a, std::uint32_t b){ return a + b;}, i , j));
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
  zinhart::parallel::thread_pool::task_future<std::int32_t> result = zinhart::parallel::default_thread_pool::push_task([&t](){return t.add_one(3);});
  ASSERT_EQ(result.get(), 4);
}
std::int32_t plus_one(std::int32_t s)
{
  return s + 1;
}
TEST(thread_pool, call_add_task_function)
{
  zinhart::parallel::thread_pool::task_future<std::int32_t> result = zinhart::parallel::default_thread_pool::push_task(plus_one,3);
  ASSERT_EQ(result.get(), 4);
}
