#include "concurrent_routines/concurrent_routines.hh"
#include "gtest/gtest.h"
#include <iostream>
#include <random>
#include <limits>
#include <memory>
//using namespace zinhart;
TEST(cpu_test, paralell_saxpy_cpu)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<double> real_dist(std::numeric_limits<double>::min(), std::numeric_limits<double>::max());
  double alpha = real_dist(mt);
  std::uint32_t n_elements = uint_dist(mt);
  std::shared_ptr<double> x_parallel = std::shared_ptr<double>(new double [n_elements]);
  std::shared_ptr<double> y_parallel = std::shared_ptr<double>(new double [n_elements]);
  std::shared_ptr<double> x_serial = std::shared_ptr<double>(new double [n_elements]);
  std::shared_ptr<double> y_serial = std::shared_ptr<double>(new double [n_elements]);
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
	double first = real_dist(mt);
	double second = real_dist(mt);
	x_serial.get()[i] = first;
	y_serial.get()[i] = second;
	x_parallel.get()[i] = first;
	y_parallel.get()[i] = second;
  }
  auto serial_saxpy = [](
						const double & a, double * x, double * y,
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
