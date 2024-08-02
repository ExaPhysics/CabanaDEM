#include <fstream>
#include <iostream>
#include <math.h>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaDEM.hpp>

#define DIM 3

// Simulate elastic wave propagation from an initial displacement field.
// void SphereSphereCollision01( const std::string filename )
void SphereSphereCollision01()
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana Rigid body solver example\n" << std::endl;


  // ====================================================
  //             Use default Kokkos spaces
  // ====================================================
  // using exec_space = Kokkos::DefaultExecutionSpace;
  // using memory_space = typename exec_space::memory_space;
  using exec_space = Kokkos::OpenMP;
  using memory_space = typename exec_space::memory_space;

  // // ====================================================
  // //                   Read inputs
  // // ====================================================
  // CabanaDEM::Inputs inputs( filename );

  // // ====================================================
  // //                Material parameters
  // // ====================================================
  // double rho0 = inputs["density"];
  // auto K = inputs["bulk_modulus"];
  // double G = inputs["shear_modulus"];
  // double delta = inputs["horizon"];
  // delta += 1e-10;

  // // ====================================================
  // //                  Discretization
  // // ====================================================
  // // FIXME: set halo width based on delta
  // std::array<double, 3> low_corner = inputs["low_corner"];
  // std::array<double, 3> high_corner = inputs["high_corner"];
  // std::array<int, 3> num_cells = inputs["num_cells"];
  // int m = std::floor( delta /
  //                     ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
  // int halo_width = m + 1; // Just to be safe.

  // // ====================================================
  // //                    Force model
  // // ====================================================
  // using model_type =
  //   CabanaDEM::ForceModel<CabanaDEM::LinearLPS, CabanaDEM::Elastic>;
  // model_type force_model( delta, K, G );

  // ====================================================
  //                 Particle generation
  // ====================================================
  // Does not set displacements, velocities, etc.
  auto particles = std::make_shared<
    CabanaDEM::Particles<memory_space, DIM>>(exec_space(), 10);

  // ====================================================
  //            Custom particle initialization
  // ====================================================
  auto x = particles->slicePosition();
  auto u = particles->sliceVelocity();
  auto m = particles->sliceMass();
  auto rho = particles->sliceDensity();
  auto rad = particles->sliceRadius();

  auto init_functor = KOKKOS_LAMBDA( const int pid )
      {
        // Initial conditions: displacements and velocities
        double rho_i = 2000.;
        double radius_i = 0.1;
        double m_i = 4. / 3. * M_PI * radius_i * radius_i * radius_i * rho_i;
	x( pid, 0 ) = -radius_i + pid * (2. * radius_i) + radius_i / 10.;
	x( pid, 1 ) = 0.;
	x( pid, 2 ) = 0.;
	u( pid, 0 ) = 0.0;
	u( pid, 1 ) = 0.0;
	u( pid, 2 ) = 0.0;

        m( pid ) = m_i;
        rho( pid ) = rho_i;
        rad( pid ) = radius_i;

      };
  particles->updateParticles( exec_space{}, init_functor );
  particles->output( 0, 1. );
  particles->output( 10, 1.1 );
  particles->output( 30, 1.1 );

  // // ====================================================
  // //                   Create solver
    // // ====================================================
    // auto cabana_dem = CabanaDEM::createSolverElastic<memory_space>( inputs, particles, force_model );

    // // ====================================================
    // //                   Simulation run
    // // ====================================================
    // cabana_dem->init();
    // cabana_dem->run();

    // // ====================================================
    // //                      Outputs
    // // ====================================================
    // // Output displacement along the x-axis
    // createDisplacementProfile( MPI_COMM_WORLD, num_cells[0], 0,
    //                            "displacement_profile.txt", *particles );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    // 01SphereSphereCollision( argv[1] );
    SphereSphereCollision01();

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
