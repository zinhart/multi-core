#include "concurrent_routines/concurrent_routines.hh"
#include "gtest/gtest.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <limits>
#include <memory>
#include <functional>

TEST(cpu_test, paralell_saxpy)
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
  zinhart::paralell_saxpy(alpha, x_parallel.get(), y_parallel.get(), n_elements);
  serial_saxpy(alpha, x_serial.get(), y_serial.get(), n_elements);
  for(i = 0; i < n_elements; ++i)
  {
		ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
  std::cout<<"Hello From CPU Tests\n";
}

TEST(cpu_test, paralell_copy)
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
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial.get()[i] = first;
	x_parallel.get()[i] = first;
  }
  zinhart::paralell_copy(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get() );
  std::copy(x_serial.get(), x_serial.get() + n_elements, y_serial.get());
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
}

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
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
		float first = real_dist(mt);
		x_serial.get()[i] = first;
		x_parallel.get()[i] = first;
  }
	auto unary_predicate = [](float & init){ return (init >= 1.0) ? true : false ;};
  zinhart::paralell_copy_if(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), unary_predicate );
  std::copy_if(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), unary_predicate);
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
  zinhart::parallel_replace(x_parallel.get(), x_parallel.get() + n_elements, old_value, new_value);
  std::replace(x_serial.get(), x_serial.get() + n_elements, old_value, new_value);
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
  zinhart::parallel_replace_if(x_parallel.get(), x_parallel.get() + n_elements, unary_predicate, new_value);
  std::replace_if(x_serial.get(), x_serial.get() + n_elements, unary_predicate, new_value);
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
  std::uint32_t i = 0;
  float new_value = real_dist(mt);
  float old_value = real_dist(mt);
  for(i = 0; i < n_elements; ++i )
  {
	x_serial.get()[i] = old_value;
	x_parallel.get()[i] = old_value;
  }
  zinhart::parallel_replace_copy(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), old_value, new_value );
  std::replace_copy(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), old_value, new_value);
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
  zinhart::parallel_replace_copy_if(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), unary_predicate, new_value);
  std::replace_copy_if(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), unary_predicate, new_value);
  //double check we have the new value 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
}

/*
TEST(cpu_test, parallel_sample)
{


}
*/

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
  parallel_ret =  zinhart::parallel_inner_product(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), init);
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
  parallel_ret =  zinhart::parallel_inner_product(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(), init, std::plus<float>(), std::equal_to<float>());
  serial_ret = std::inner_product(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), init,
	  std::plus<float>(), std::equal_to<float>());
  ASSERT_EQ(parallel_ret, serial_ret);
}
/*
//to do
TEST(cpu_test, parallel_transform_reduce)
{

}
*/

TEST(cpu_test, paralell_accumulate)
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
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial.get()[i] = first;
	x_parallel.get()[i] = first;
  }
  //sum
  float p_sum = zinhart::paralell_accumalute(x_parallel.get(), x_parallel.get() + n_elements, 0 );
  float s_sum = std::accumulate(x_serial.get(), x_serial.get() + n_elements, 0);
  //double check we have the same values 
  ASSERT_EQ(p_sum,s_sum);
}

TEST(cpu_test, paralell_for_each)
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
  zinhart::paralell_for_each(x_parallel.get(), x_parallel.get() + n_elements, unary);
  std::for_each(x_serial.get(), x_serial.get() + n_elements, unary);
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel.get()[i], x_serial.get()[i]);
  }
}


TEST(cpu_test, paralell_transform)
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
  zinhart::paralell_transform(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get(),unary );
  std::transform(x_serial.get(), x_serial.get() + n_elements, y_serial.get(), unary);
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
}

TEST(cpu_test, paralell_generate)
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
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial.get()[i] = first;
	x_parallel.get()[i] = first;
  }
  auto generator = [](){ return -2.0; };
  zinhart::paralell_generate(x_parallel.get(), x_parallel.get() + n_elements, generator);
  std::generate(x_serial.get(), x_serial.get() + n_elements, generator);
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel.get()[i], x_serial.get()[i]);
  }
}
