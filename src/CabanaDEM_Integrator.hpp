#ifndef CabanaDEM_TIMEINTEGRATOR_HPP
#define CabanaDEM_TIMEINTEGRATOR_HPP

#include <memory>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaDEM
{
  template <class ExecutionSpace>
  class Integrator
  {
    using exec_space = ExecutionSpace;

    double _dt, _half_dt;
  public:
    Integrator ( double dt )
      : _dt (dt)
    {
      _half_dt = 0.5 * dt;

    }
    ~Integrator() {}

    template <class ParticlesType>
    void stage1(ParticlesType& p){
      auto u = p.sliceVelocity();
      auto omega = p.sliceOmega();
      auto force = p.sliceForce();
      auto torque = p.sliceTorque();
      auto m = p.sliceMass();
      auto I = p.sliceMomentOfInertia();
      auto dt = _dt;
      auto half_dt = _half_dt;
      auto dem_stage1_func = KOKKOS_LAMBDA( const int i )
	{
	  double m_inverse = 1. / m ( i );

	  u( i, 0 ) += _half_dt * force( i, 0 ) * m_inverse;
	  u( i, 1 ) += _half_dt * force( i, 1 ) * m_inverse;
	  u( i, 2 ) += _half_dt * force( i, 2 ) * m_inverse;

	  double I_inverse = 1. / I ( i );
	  omega( i, 0 ) += _half_dt * torque( i, 0 ) * I_inverse;
	  omega( i, 1 ) += _half_dt * torque( i, 1 ) * I_inverse;
	  omega( i, 2 ) += _half_dt * torque( i, 2 ) * I_inverse;
	};
      Kokkos::RangePolicy<exec_space> policy( 0, u.size() );
      Kokkos::parallel_for( "CabanaPD::Integrator::Stage1", policy,
			    dem_stage1_func );
    }

    template <class ParticlesType>
    void stage2(ParticlesType& p){
      // _time.start();
      auto x = p.slicePosition();
      auto u = p.sliceVelocity();
      auto dt = _dt;
      auto half_dt = _half_dt;
      auto dem_stage2_func = KOKKOS_LAMBDA( const int i )
	{
	  x( i, 0 ) += dt * u( i, 0 );
	  x( i, 1 ) += dt * u( i, 1 );
	  x( i, 2 ) += dt * u( i, 2 );
	};
      Kokkos::RangePolicy<exec_space> policy( 0, u.size() );
      Kokkos::parallel_for( "CabanaPD::Integrator::Stage2", policy,
			    dem_stage2_func );
    }

    template <class ParticlesType>
    void stage3(ParticlesType& p){
      // _time.start();
      auto u = p.sliceVelocity();
      auto omega = p.sliceOmega();
      auto force = p.sliceForce();
      auto torque = p.sliceTorque();
      auto m = p.sliceMass();
      auto I = p.sliceMomentOfInertia();
      auto dt = _dt;
      auto half_dt = _half_dt;
      auto dem_stage3_func = KOKKOS_LAMBDA( const int i )
	{
	  double m_inverse = 1. / m ( i );

	  u( i, 0 ) += _half_dt * force( i, 0 ) * m_inverse;
	  u( i, 1 ) += _half_dt * force( i, 1 ) * m_inverse;
	  u( i, 2 ) += _half_dt * force( i, 2 ) * m_inverse;

	  double I_inverse = 1. / I ( i );
	  omega( i, 0 ) += _half_dt * torque( i, 0 ) * I_inverse;
	  omega( i, 1 ) += _half_dt * torque( i, 1 ) * I_inverse;
	  omega( i, 2 ) += _half_dt * torque( i, 2 ) * I_inverse;
	};
      Kokkos::RangePolicy<exec_space> policy( 0, u.size() );
      Kokkos::parallel_for( "CabanaPD::Integrator::Stage3", policy,
			    dem_stage3_func );
    }

  };
}

#endif // EXAMPM_TIMEINTEGRATOR_HPP
