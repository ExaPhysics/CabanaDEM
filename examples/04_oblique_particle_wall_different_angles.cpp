#include <fstream>
#include <iostream>
#include <math.h>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaDEM.hpp>

#define DIM 3

// Simulate two spherical particles colliding head on
void SphereSphereCollision01( const std::string filename )
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana Rigid body solver example\n" << std::endl;


  // ====================================================
  //             Use default Kokkos spaces
  // ====================================================
  using exec_space = Kokkos::DefaultExecutionSpace;
  using memory_space = typename exec_space::memory_space;
  // using exec_space = Kokkos::OpenMP;
  // using memory_space = typename exec_space::memory_space;

  // ====================================================
  //                   Read inputs
  // ====================================================
  CabanaDEM::Inputs inputs( filename );

  // // ====================================================
  // //                Material parameters
  // // ====================================================
  double rho_i = inputs["density"];
  auto E_i = inputs["youngs_modulus"];
  double G_i = inputs["shear_modulus"];
  double nu_i = inputs["poissons_ratio"];
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

  // ====================================================
  //                    Force model
  // ====================================================
  // using model_type =
  //   CabanaDEM::Force<exec_space>;
  // model_type force_model();

  // ====================================================
  //                 Particle generation
  // ====================================================
  // Does not set displacements, velocities, etc.
  auto particles = std::make_shared<
    CabanaDEM::Particles<memory_space, DIM>>(exec_space(), 2);

  // ====================================================
  //            Custom particle initialization
  // ====================================================
  auto x = particles->slicePosition();
  auto u = particles->sliceVelocity();
  auto m = particles->sliceMass();
  auto rho = particles->sliceDensity();
  auto rad = particles->sliceRadius();
  auto I = particles->sliceMomentOfInertia();
  auto E = particles->sliceYoungsMod();
  auto nu = particles->slicePoissonsRatio();
  auto G = particles->sliceShearMod();

  auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Initial conditions: displacements and velocities
        double rho_i = 2000.;
        double radius_i = 0.1;
        double m_i = 4. / 3. * M_PI * radius_i * radius_i * radius_i * rho_i;
        double I_i = 2. / 5. * m_i * radius_i * radius_i;
	x( pid, 0 ) = -radius_i + pid * (2. * radius_i) + radius_i / 10.;
	x( pid, 1 ) = 0.;
	x( pid, 2 ) = 0.;
	u( pid, 0 ) = 1.0;
	if (pid == 1) u( pid, 0 ) = -1.0;
	u( pid, 1 ) = 0.0;
	u( pid, 2 ) = 0.0;

        m( pid ) = m_i;
        I( pid ) = I_i;
        rho( pid ) = rho_i;
        rad( pid ) = radius_i;
        E( pid ) = E_i;
        G( pid ) = G_i;
        nu( pid ) = nu_i;
      };
  particles->updateParticles( exec_space{}, init_functor );

  // ====================================================
  //                 Set the neighbours mesh limits
  // ====================================================
  particles->mesh_lo[0] = -4. * 0.1;
  particles->mesh_lo[1] = -4. * 0.1;
  particles->mesh_lo[2] = -4. * 0.1;

  particles->mesh_hi[0] = 4. * 0.1;
  particles->mesh_hi[1] = 4. * 0.1;
  particles->mesh_hi[2] = 4. * 0.1;


  // ====================================================
  //                   Create solver
  // ====================================================
  auto cabana_dem = CabanaDEM::createSolverDEM<memory_space>( inputs, particles );

  // ====================================================
  //                   Simulation run
  // ====================================================
  // cabana_dem->init();
  cabana_dem->run();
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    SphereSphereCollision01( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
