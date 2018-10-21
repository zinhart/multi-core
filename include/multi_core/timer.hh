#ifndef TIMER_HH
#define TIMER_HH
#include <chrono>
#include <iostream>
namespace zinhart
{
  template<class TimeUnit>
  class timer 
  {
	typedef std::chrono::high_resolution_clock high_resolution_clock;
	private:
	high_resolution_clock::time_point start;
	public:
	explicit timer(bool run = false)
	{
	  if (run)
		set();
	}
	void set()
	{
	  start = high_resolution_clock::now();
	}
	TimeUnit elapsed() const
	{
	  return std::chrono::duration_cast<TimeUnit>(high_resolution_clock::now() - start);
	}
	template <typename T, typename Traits>
	  friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, const timer& t)
	  {
		return out << t.elapsed().count();	
	  }
  };
}
#endif
