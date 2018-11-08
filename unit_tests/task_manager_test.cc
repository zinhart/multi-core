#include <multi_core/multi_core.hh>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <limits>
#include <chrono>
using namespace testing;


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

TEST(task_manager, push_wait)
{
  example ret;
  zinhart::multi_core::task_manager<example> t;
  ret = t.push_wait(0, [](char a)
	{
	  example x; 
	  x.set_uchar(a);
	  return x;
	}, 'a'
  );
  ASSERT_EQ(ret.get_uchar(), 'a');

  ret = t.push_wait(0, [](std::uint32_t num)
	{
	  example x; 
	  x.set_uint(num);
	  return x;
	}, 1
  );
  ASSERT_EQ(ret.get_uint(), 1);

  ret = t.push_wait(0, [](std::string s)
	{
	  example x; 
	  x.set_string(s);
	  return x;
	}, "apples"
  );
  ASSERT_EQ(ret.get_string(), "apples");
}

TEST(task_manager, push)
{
  zinhart::multi_core::task_manager<example> t;
  t.push(0, [](char a)
	{
	  example x; 
	  x.set_uchar(a);
	  return x;
	}, 'a'
  );
  ASSERT_EQ(t.get(0).get_uchar(), 'a');
  
  t.push(0, [](std::uint32_t num)
	{
	  example x; 
	  x.set_uint(num);
	  return x;
	},
	1
  );
  ASSERT_EQ(t.get(1).get_uint(), 1);

  t.push(0, [](std::string s)
	{
	  example x; 
	  x.set_string(s);
	  return x;
	}, "apples"
  );
 example p(t.get(2));
 ASSERT_EQ(p.get_string(), "apples"); 
}
