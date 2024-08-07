#ifndef CabanaDEMWall_HPP
#define CabanaDEMWall_HPP

#include <memory>
#include <filesystem> // or #include <filesystem> for C++17 and up

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>


namespace fs = std::filesystem;


#define DIM 3

namespace CabanaDEM
{
  template <class MemorySpace, int Dimension>
  class Wall
  {
  public:
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;
    static constexpr int dim = Dimension;

    using double_type = Cabana::MemberTypes<double>;
    using int_type = Cabana::MemberTypes<int>;
    using vec_double_type = Cabana::MemberTypes<double[dim]>;
    using vec_int_type = Cabana::MemberTypes<int[dim]>;

    // FIXME: add vector length.
    // FIXME: enable variable aosoa.
    using aosoa_double_type = Cabana::AoSoA<double_type, memory_space, 1>;
    using aosoa_int_type = Cabana::AoSoA<int_type, memory_space, 1>;
    using aosoa_vec_double_type = Cabana::AoSoA<vec_double_type, memory_space, 1>;
    using aosoa_vec_int_type = Cabana::AoSoA<vec_int_type, memory_space, 1>;

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    Wall( const ExecSpace& exec_space, std::size_t no_of_walls, std::string output_folder_name )
    {
      _no_of_walls = no_of_walls;
      _output_folder_name = output_folder_name;
      // create the output folder if it doesn't exist

      if (!fs::is_directory(_output_folder_name) || !fs::exists(_output_folder_name)) { // Check if src folder exists
	fs::create_directory(_output_folder_name); // create src folder
      }

      resize( _no_of_walls );
      createParticles( exec_space );
    }

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    void createParticles( const ExecSpace& exec_space )
    {
      auto x = slicePosition();

      auto create_particles_func = KOKKOS_LAMBDA( const int i )
	{
	  for (int j=0; j < DIM; j++){
	    // x( i, j ) = DIM * i + j;
	  }
	};
      Kokkos::RangePolicy<ExecSpace> policy( 0, x.size() );
      Kokkos::parallel_for( "create_particles_lambda", policy,
			    create_particles_func );
    }

    template <class ExecSpace, class FunctorType>
    void updateParticles( const ExecSpace, const FunctorType init_functor )
    {
      Kokkos::RangePolicy<ExecSpace> policy( 0, _no_of_walls );
      Kokkos::parallel_for(
			   "CabanaPD::Particles::update_particles", policy,
			   KOKKOS_LAMBDA( const int pid ) { init_functor( pid ); } );
    }

    auto slicePosition()
    {
      return Cabana::slice<0>( _x, "positions" );
    }
    auto slicePosition() const
    {
      return Cabana::slice<0>( _x, "positions" );
    }

    auto sliceVelocity()
    {
      return Cabana::slice<0>( _u, "velocities" );
    }
    auto sliceVelocity() const
    {
      return Cabana::slice<0>( _u, "velocities" );
    }

    auto sliceOmega()
    {
      return Cabana::slice<0>( _omega, "omega" );
    }
    auto sliceOmega() const
    {
      return Cabana::slice<0>( _omega, "omega" );
    }

    auto sliceYoungsMod() {
      return Cabana::slice<0>( _E, "youngs_mod" );
    }
    auto sliceYoungsMod() const
    {
      return Cabana::slice<0>( _E, "youngs_mod" );
    }

    auto slicePoissonsRatio() {
      return Cabana::slice<0>( _nu, "poissons_ratio" );
    }
    auto slicePoissonsRatio() const
    {
      return Cabana::slice<0>( _nu, "poissons_ratio" );
    }

    auto sliceShearMod() {
      return Cabana::slice<0>( _G, "shear_mod" );
    }
    auto sliceShearMod() const
    {
      return Cabana::slice<0>( _G, "shear_mod" );
    }

    auto sliceNormal()
    {
      return Cabana::slice<0>( _normal, "normal" );
    }
    auto sliceNormal() const
    {
      return Cabana::slice<0>( _normal, "normal" );
    }

    void resize(const std::size_t n)
    {
      _x.resize( n );
      _u.resize( n );
      _normal.resize( n );
      _omega.resize( n );
      _E.resize( n );
      _nu.resize( n );
      _G.resize( n );
    }

    void output(  const int output_step,
                  const double output_time,
                  const bool use_reference = true )
    {
      // _output_timer.start();

#ifdef Cabana_ENABLE_HDF5
      Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
							      h5_config,
							      // _output_folder_name+"/particles",
							      "wall",
							      MPI_COMM_WORLD,
							      output_step,
							      output_time,
							      _no_of_walls,
							      slicePosition());
      // #else
      // #ifdef Cabana_ENABLE_SILO
      //       Cabana::Grid::Experimental::SiloParticleOutput::
      // 	writePartialRangeTimeStep(
      // 				  "particles", output_step, output_time,
      // 				  _no_of_particles,
      // 				  slicePosition(),
      // 				  sliceVelocity(),
      // 				  sliceAcceleration(),
      // 				  sliceMass(),
      // 				  sliceDensity(),
      // 				  sliceRadius());
#else
      std::cout << "No particle output enabled.";
      // log( std::cout, "No particle output enabled." );
      // #endif
#endif

      // _output_timer.stop();
    }

  private:
    int _no_of_walls;
    aosoa_vec_double_type _x;
    aosoa_vec_double_type _u;
    aosoa_vec_double_type _omega;
    aosoa_vec_double_type _normal;
    aosoa_double_type _E;
    aosoa_double_type _nu;
    aosoa_double_type _G;
    aosoa_double_type _I;
    std::string _output_folder_name;

#ifdef Cabana_ENABLE_HDF5
    Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
#endif

    // Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
  };

} // namespace CabanaDEM

#endif
