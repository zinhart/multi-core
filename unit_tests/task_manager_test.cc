#include <multi_core/multi_core.hh>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <limits>
#include <chrono>
using namespace testing;
TEST(task_manager, push_wait)
{
  union test
  {
	std::int32_t i32;
	std::uint32_t ui32;
	float  f;
	double d;
  };
  zinhart::multi_core::task_manager<test> t;
  t.push_wait(0, [](){ std::cout<<"apples\n";});
 // test x.i32 = t.push_wait(0, []()->int{return 1;});
 // ASSERT_EQ(x.i32, 1);
}

TEST(task_manager, push)
{
  union test
  {
	std::uint32_t ui32;
	std::uint8_t ui8;
  };
  zinhart::multi_core::task_manager<test> t;
  t.push(0, [](){test x; x.ui32 = 1; return x;});
  ASSERT_EQ(t.get(0).ui32, 1);

  t.push(0, [](){test x; x.ui8 = 'a'; return x;});
  ASSERT_EQ(t.get(1).ui8, 'a');

 /* t.push(0,[](){std::cout<<"apples\n";test x; x.i32 = 1; return x;});
  t.get(2);*/
}
