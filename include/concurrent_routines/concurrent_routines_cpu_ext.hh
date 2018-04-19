#ifndef CONCURRENT_ROUTINES_CPU_EXT_HH
#define CONCURRENT_ROUTINES_CPU_EXT_HH
namespace zinhart
{
  //CPU ROUTINES
  // will be used for specialization of saxypy, for each, accumulate, etc.
  // will default to stl implementations on sequatial because why reinvent the wheel
  enum class EXCECUTION_POLICY : std::uint8_t {SEQUENTIAL = 0, PARALLEL = 1};
  
  template <EXCECUTION_POLICY>
	class copy;
  template<>
	class copy<EXCECUTION_POLICY::SEQUENTIAL>;
  template<>
	class copy<EXCECUTION_POLICY::PARALLEL>;

  template <EXCECUTION_POLICY>
	class accumulate;
  template<>
	class accumulate<EXCECUTION_POLICY::SEQUENTIAL>;
  template<>
	class accumulate<EXCECUTION_POLICY::PARALLEL>;

  template <EXCECUTION_POLICY>
	class for_each;
  template<>
	class for_each<EXCECUTION_POLICY::SEQUENTIAL>;
  template<>
	class for_each<EXCECUTION_POLICY::PARALLEL>;

  template <EXCECUTION_POLICY>
	class transform;
  template<>
	class transform<EXCECUTION_POLICY::SEQUENTIAL>;
  template<>
	class transform<EXCECUTION_POLICY::PARALLEL>;

  template <EXCECUTION_POLICY>
	class generate;
  template<>
	class generate<EXCECUTION_POLICY::SEQUENTIAL>;
  template<>
	class generate<EXCECUTION_POLICY::PARALLEL>;
  
  template <EXCECUTION_POLICY>
	class saxpy;
  template<>
	class saxpy<EXCECUTION_POLICY::SEQUENTIAL>;
  template<>
	class saxpy<EXCECUTION_POLICY::PARALLEL>;


}
#include "../src/cpu.tcc"
#endif

