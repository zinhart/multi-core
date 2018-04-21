#include "concurrent_routines/concurrent_routines.hh"
#include "gtest/gtest.h"
#include <iostream>
#include <random>
#include <limits>
#include <memory>

TEST(cpu_test, paralell_saxpy_cpu)
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
  zinhart::paralell_saxpy_cpu(alpha, x_parallel.get(), y_parallel.get(), n_elements);
  serial_saxpy(alpha, x_serial.get(), y_serial.get(), n_elements);
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
  std::cout<<"Hello From CPU Tests\n";
}

TEST(cpu_test, paralell_copy_cpu)
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
  zinhart::paralell_copy_cpu(x_parallel.get(), x_parallel.get() + n_elements, y_parallel.get() );
  std::copy(x_serial.get(), x_serial.get() + n_elements, y_serial.get());
  //double check we have the same values 
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(y_parallel.get()[i], y_serial.get()[i]);
  }
}

TEST(cpu_test, paralell_accumulate_cpu)
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
  float p_sum = zinhart::paralell_accumalute_cpu(x_parallel.get(), x_parallel.get() + n_elements, 0 );
  float s_sum = std::accumulate(x_serial.get(), x_serial.get() + n_elements, 0);
  //double check we have the same values 
  ASSERT_EQ(p_sum,s_sum);
}

TEST(cpu_test, paralell_for_each_cpu)
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
  std::shared_ptr<float> x_serial = std::shared_ptr<float>(new float [n_elements]);
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	float first = real_dist(mt);
	x_serial.get()[i] = first;
	x_parallel.get()[i] = first;
  }
  auto unary = []( const float & a )
						{
						  a = a * 2.0;
						};
  zinhart::paralell_for_each_cpu(x_parallel.get(), x_parallel.get() + n_elements, unary);
  std::for_each(x_serial.get(), x_serial.get() + n_elements, unary);
  for(i = 0; i < n_elements; ++i)
  {
	ASSERT_EQ(x_parallel.get()[i], x_serial.get()[i]);
  }
}
