#ifndef CabanaDEM_TIMEINTEGRATOR_HPP
#define CabanaDEM_TIMEINTEGRATOR_HPP

#include <memory>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaDEM
{
  namespace TimeIntegrator
  {

    template <class ExecutionSpace>
    class Integrator
    {
      using exec_space = ExecutionSpace;

      double _dt, _half_dt;
      // Timer _timer;
    public:
      Integrator ( double dt )
	: _dt (dt)
      {
	_half_dt = 0.5 * dt;

      }
      ~Integrator() {}

      template <class ParticlesType>
      void initialHalfStep(ParticlesType& p){
	// _time.start();
	auto x = p.slicePosition();
	auto dt = _dt;
	auto half_dt = _half_dt;
	auto init_func = KOKKOS_LAMBDA( const int i )
	  {
	    x( i, 0 ) += dt * 1.;
	    x( i, 1 ) += dt * 1.;
	    x( i, 2 ) += dt * 1.;
	  };
	Kokkos::RangePolicy<exec_space> policy( 0, x.size() );
	Kokkos::parallel_for( "CabanaPD::Integrator::Initial", policy,
			      init_func );

	// _timer.stop();
      }

      double timeInit() { return 0.0; };
      // auto time() { return _timer.time(); };

    };
  }
}

#endif // EXAMPM_TIMEINTEGRATOR_HPP
