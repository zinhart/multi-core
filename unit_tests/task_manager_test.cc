#include <multi_core/multi_core.hh>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <limits>
#include <chrono>
using namespace testing;
TEST(task_manager, push_wait)
{
  /*
  union test
  {
	std::int32_t i32;
	std::uint32_t ui32;
	float  f;
	double d;
  };
  zinhart::multi_core::task_manager<test> t;
  t.push_wait(0, [](){ std::cout<<"apples\n";});
  test x.i32 = t.push_wait(0, []()->int{return 1;});
  ASSERT_EQ(x.i32, 1);
  */
}

TEST(task_manager, push)
{
  union test
  {

	private:
      enum class tags{uchar, uint, string} tag;
	  std::uint8_t ui8;
	  std::uint32_t ui32;
      std::string str;
	public:

	test()
	{
	  ui8 = 0;
	  ui32 = 0;
	}
	test(const test & t)
	{
	  if(t.get_tag() == tags::uchar)
	  {
		ui8 = t.get_uchar();
	  }
	  else if(t.get_tag() == tags::uint)
	  {
		ui32 = t.get_uint();
	  }
	  else if(t.get_tag() == tags::string)
	  {
		set_string(t.get_string());
	  }

	  set_tag(t.get_tag());
	}
	
	explicit test(std::uint8_t x)
	{
	  ui8 = x;
	  tag = tags::uchar;
	}

	explicit test(std::uint32_t x)
	{
	  ui8 = x;
	  tag = tags::uint;
	}
	void set_uchar(std::uint8_t x)
	{
	  tag = tags::uchar;
	  ui8 = x;
	}

	void set_uint(std::uint32_t x)
	{
	  tag = tags::uint;
	  ui32 = x;
	}
	void set_string(std::string s)
	{
	  tag = tags::string;
	  new (&str) std::string(s);// here
	}
	std::uint8_t get_uchar()const
	{
	  return ui8;
	}

	std::uint32_t get_uint()const
	{
	  return ui32;
	}
	std::string get_string()const
	{
	  return str;
	}

	tags get_tag()const
	{
	  return tag;
	}
	void set_tag(tags tag)
	{
	  this->tag = tag;
	}

	test & operator = (const test & t)
	{

	  if(t.get_tag() == tags::uchar)
	  {
		ui8 = t.get_uchar();
	  }
	  else if(t.get_tag() == tags::uint)
	  {
		ui32 = t.get_uint();
	  }
	  else if(t.get_tag() == tags::string)
	  {
		set_string(t.get_string());
	  }
	  return *this;
	}
/*
	explicit test(std::string x)
	{
	  str = x;
	}

	test(const test & x)
	{
	  str =  x.str;
	}
*/

	~test()
	{
	  if(get_tag() == tags::string)
		str.~basic_string();
	}


  };
  
  zinhart::multi_core::task_manager<test> t;

  t.push(0, []()
	{
	  test x;//('a'); 
	  x.set_uchar('a');
	  return x;
	}
  );
  ASSERT_EQ(t.get(0).get_uchar(), 'a');
  
  t.push(0, []()
	{
	  test x;//(1); 
	  x.set_uint(1);
	  return x;}
  );
  ASSERT_EQ(t.get(1).get_uint(), 1);

  t.push(0, []()
	{
	  test x;//(1); 
	  x.set_string("apples");
	  return x;
	}
  );
//  ASSERT_EQ(t.get(2).get_string(), "apples");

/*
  t.push(0, []()->test{test x{"apples"}; x.str = "apples"; return x;});
  ASSERT_EQ(t.get(2).str, "apples");
  std::cout<<"here"<<"\n";
*/
 /* t.push(0,[](){std::cout<<"apples\n";test x; x.i32 = 1; return x;});
  t.get(2);*/
}
