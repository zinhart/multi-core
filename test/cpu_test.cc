#include "concurrent_routines/concurrent_routines.hh"
#include "gtest/gtest.h"
#include <iostream>
#include <random>
#include <limits>
#include <memory>
//using namespace zinhart;
TEST(cpu_test, launch_cpu_threaded_saxpy)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  //for any needed random uint
  std::uniform_int_distribution<std::uint16_t> uint_dist(1, std::numeric_limits<std::uint16_t>::max());
  //for any needed random real
  std::uniform_real_distribution<double> real_dist(std::numeric_limits<double>::min(), std::numeric_limits<double>::max());
  double alpha = real_dist(mt);
  std::uint32_t n_elements = 40;//uint_dist(mt);
  std::shared_ptr<double> x = std::make_shared<double>(n_elements); 
  std::shared_ptr<double> y = std::make_shared<double>(n_elements);
  std::uint32_t i = 0;
  for(i = 0; i < n_elements; ++i )
  {
   // weird memory corruption here   
   // x.get()[i] = real_dist(mt);
   // y.get()[i] = real_dist(mt);
  }
  zinhart::launch_cpu_threaded_saxpy(alpha, x.get(), y.get(), n_elements);
  std::cout<<"Hello From CPU Tests\n";
}
