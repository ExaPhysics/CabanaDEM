/*
  How to run this examples:
  ./examples/04ObliqueParticleWallDifferentAngles ../examples/inputs/04_oblique_particle_wall_different_angles.json ./

*/
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaDEM.hpp>

#define DIM 3


// Simulate two spherical particles colliding head on
double ObliqueParticleWallDifferentAngles04( const std::string input_filename,
					     const std::string output_folder_name,
					     const std::string incident_angle_input)
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

  // ====================================================
  //                   Read inputs
  // ====================================================
  CabanaDEM::Inputs inputs( input_filename );
  // also decode some of the inputs
  char delimiter = '=';
  std::vector<std::string> result = CabanaDEM::splitString(incident_angle_input, delimiter);

  // double incident_angle = 20.;
  double incident_angle = std::atof(result[1].c_str());


  // ====================================================
  //                Material parameters
  // ====================================================
  // Particle material properties
  double rho_p_inp = inputs["particle_density"];
  double radius_p_inp = inputs["particle_radius"];
  double  E_p_inp = inputs["particle_youngs_modulus"];
  double G_p_inp = inputs["particle_shear_modulus"];
  double nu_p_inp = inputs["particle_poissons_ratio"];
  // Wall material properties
  double  E_w_inp = inputs["wall_youngs_modulus"];
  double G_w_inp = inputs["wall_shear_modulus"];
  double nu_w_inp = inputs["wall_poissons_ratio"];
  // Coefficient of restitution among the interacting bodies
  double cor_pp_inp = inputs["coefficient_of_restitution_pp"];
  double cor_pw_inp = inputs["coefficient_of_restitution_pw"];
  // friction among the interacting bodies
  double friction_pp_inp = inputs["friction_pp"];
  double friction_pw_inp = inputs["friction_pw"];

  // ====================================================
  //                Geometric properties
  // ====================================================
  double velocity_p_inp = inputs["velocity_p"];

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
  //  Force model
  // ====================================================
  auto force = std::make_shared<
    CabanaDEM::Force<exec_space>>(cor_pp_inp, cor_pw_inp,
				  friction_pp_inp, friction_pw_inp);

  // ====================================================
  //                 Particle generation
  // ====================================================
  // Does not set displacements, velocities, etc.
  auto particles = std::make_shared<
    CabanaDEM::Particles<memory_space, DIM, 6, 1>>(exec_space(), 1, output_folder_name);

  // ====================================================
  //            Custom particle initialization
  // ====================================================
  auto x_p = particles->slicePosition();
  auto u_p = particles->sliceVelocity();
  auto omega_p = particles->sliceOmega();
  auto m_p = particles->sliceMass();
  auto rho_p = particles->sliceDensity();
  auto rad_p = particles->sliceRadius();
  auto I_p = particles->sliceMomentOfInertia();
  auto E_p = particles->sliceYoungsMod();
  auto nu_p = particles->slicePoissonsRatio();
  auto G_p = particles->sliceShearMod();
  std::cout << &u_p << std::endl;

  double angle = incident_angle / 180. * M_PI;

  double ux_p_inp = velocity_p_inp * sin(angle);
  double uy_p_inp = -velocity_p_inp * cos(angle);

  auto particles_init_functor = KOKKOS_LAMBDA( const int pid )
    {
      // Initial conditions: displacements and velocities
      double m_p_i = 4. / 3. * M_PI * radius_p_inp * radius_p_inp * radius_p_inp * rho_p_inp;
      double I_p_i = 2. / 5. * m_p_i * radius_p_inp * radius_p_inp;
      x_p( pid, 0 ) = 0.;
      x_p( pid, 1 ) = radius_p_inp + radius_p_inp / 1000000.;
      x_p( pid, 2 ) = 0.;
      u_p( pid, 0 ) = ux_p_inp;
      u_p( pid, 1 ) = uy_p_inp;
      u_p( pid, 2 ) = 0.0;

      m_p( pid ) = m_p_i;
      I_p( pid ) = I_p_i;
      rho_p( pid ) = rho_p_inp;
      rad_p( pid ) = radius_p_inp;
      E_p( pid ) = E_p_inp;
      G_p( pid ) = G_p_inp;
      nu_p( pid ) = nu_p_inp;
    };
  particles->updateParticles( exec_space{}, particles_init_functor );
  particles->output( 0, 0. );

  particles->resize(20);

  auto x_p_20 = particles->slicePosition();
  auto u_p_20 = particles->sliceVelocity();
  // std::cout << "u_p 20 is " << &u_p_20 << std::endl;

  for (int i=0; i < u_p_20.size(); i++){
    x_p_20( i, 0 ) = 0.;
    x_p_20( i, 1 ) = 3 * i * radius_p_inp + radius_p_inp / 1000000.;
    x_p_20( i, 2 ) = 0.;

    u_p_20( i, 0 ) = 0.;
    u_p_20( i, 1 ) = 1.;
    u_p_20( i, 2 ) = 0.;
  }

  // for (int i=0; i < u_p_20.size(); i++){
  //   std::cout << x_p_20 (i, 1) << "\n";
  // }

  particles->output( 10, 0.1 );

  for (int i=0; i < u_p_20.size(); i++){
    x_p_20( i, 0 )+= 2. * u_p_20( i, 0 );
    x_p_20( i, 1 )+= 2. * u_p_20( i, 1 );
    x_p_20( i, 2 )+= 2. * u_p_20( i, 2 );
  }

  particles->output( 20, 0.2 );


  particles->resize(15);

  for (int i=0; i < u_p_20.size(); i++){
    x_p_20( i, 0 ) = 0.;
    x_p_20( i, 1 ) = 5 * i * radius_p_inp + radius_p_inp / 1000000.;
    x_p_20( i, 2 ) = 0.;

    u_p_20( i, 0 ) = 0.;
    u_p_20( i, 1 ) = 1.;
    u_p_20( i, 2 ) = 0.;
  }

  particles->output( 30, 0.3 );

  return 0;

}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );
  // check inputs and write usage
  if ( argc < 3 )
    {
      std::cerr << "Usage: ./01ElasticNormalImpactOfTwoIdenticalParticles  input_file_name output_folder \n";

      std::cerr << "      input_file_name      path to input file name\n";
      std::cerr << "      output_folder        folder to save the data, example 01NormalImpact\n";
      std::cerr << "      Incident angle       incident angle, --angle=30.\n";

      Kokkos::finalize();
      MPI_Finalize();
      return 0;
    }

  ObliqueParticleWallDifferentAngles04( argv[1], argv[2], argv[3]);

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
