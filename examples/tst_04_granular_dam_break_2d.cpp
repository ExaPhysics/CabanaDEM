#include <fstream>
#include <iostream>
#include <math.h>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaDEM.hpp>

#define DIM 3


// Function to generate particles in a 2D zigzag pattern
template<typename Y>
auto create_points(double length, double height, double spacing, Y & particles) {
    std::vector<std::pair<double, double>> points;
    // Calculate the number of rows and columns based on length, height, and spacing
    int cols = static_cast<int>(length / spacing);
    int rows = static_cast<int>(height / spacing);

    for (int i = 0; i < rows; ++i) {
        // Calculate offset for zigzag pattern
        double xOffset = (i % 2 == 0) ? 0.0 : spacing / 2.0;

        for (int j = 0; j < cols; ++j) {
            double x = j * spacing + xOffset;
            double y = i * spacing;

            // Only include particles that are within the specified length and height
            if (x <= length && y <= height) {
                points.emplace_back(x, y);
            }
        }
    }

    particles->resize(points.size());

    auto x = particles->slicePosition();
    auto u = particles->sliceVelocity();

    for (int i=0; i < u.size(); i++){
      x( i, 0 ) = points[i].first;
      x( i, 1 ) = points[i].second;
      x( i, 2 ) = 0.;
    }
}


// Simulate two spherical particles colliding head on
void GranularDamBreak2D( const std::string input_filename, const std::string output_folder_name )
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "A 2d granular dam break example\n" << std::endl;


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
  CabanaDEM::Inputs inputs( input_filename );

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
    CabanaDEM::Particles<memory_space, DIM, 6, 1>>(exec_space(), 3, output_folder_name);
  create_points(1., 1., radius_p_inp * 2., particles);

  // ====================================================
  //            Custom particle initialization
  // ====================================================
  auto x_p = particles->slicePosition();
  auto u_p = particles->sliceVelocity();
  auto m_p = particles->sliceMass();
  auto rho_p = particles->sliceDensity();
  auto rad_p = particles->sliceRadius();
  auto I_p = particles->sliceMomentOfInertia();
  auto E_p = particles->sliceYoungsMod();
  auto nu_p = particles->slicePoissonsRatio();
  auto G_p = particles->sliceShearMod();
  auto tangential_disp_ss_x = particles->sliceTangentialDispSSx();
  auto tangential_disp_ss_y = particles->sliceTangentialDispSSy();
  auto tangential_disp_ss_z = particles->sliceTangentialDispSSz();
  auto tangential_disp_idx = particles->sliceTangentialDispIdx();
  auto total_no_tangential_contacts = particles->sliceTotalNoTangentialContacts();

  auto particles_init_functor = KOKKOS_LAMBDA( const int pid )
    {
      // Initial conditions: displacements and velocities
      double m_p_i = 4. / 3. * M_PI * radius_p_inp * radius_p_inp * radius_p_inp * rho_p_inp;
      double I_p_i = 2. / 5. * m_p_i * radius_p_inp * radius_p_inp;

      // x_p( pid, 0 ) = -radius_p_inp + pid * (2. * radius_p_inp) + pid * radius_p_inp / 100.;
      // x_p( pid, 1 ) = 0.;
      // if (pid == 1){
      //   x_p( pid, 0 ) = -radius_p_inp + 1 * (1.5 * radius_p_inp) + 1 * radius_p_inp / 100.;
      //   x_p( pid, 1 ) = radius_p_inp + radius_p_inp / 2.;
      // }
      // if (pid == 2){
      //   x_p( pid, 0 ) = -radius_p_inp + 1 * (1.5 * radius_p_inp) + 1 * radius_p_inp / 100.;
      //   x_p( pid, 1 ) = - radius_p_inp - radius_p_inp / 2.;
      // }
      // x_p( pid, 2 ) = 0.;

      // u_p( pid, 0 ) = 0.;
      // if (pid == 0){
      //   u_p( pid, 0 ) = velocity_p_inp;
      // }
      // u_p( pid, 1 ) = 0.0;
      // u_p( pid, 2 ) = 0.0;

      m_p( pid ) = m_p_i;
      I_p( pid ) = I_p_i;
      rho_p( pid ) = rho_p_inp;
      rad_p( pid ) = radius_p_inp;
      E_p( pid ) = E_p_inp;
      G_p( pid ) = G_p_inp;
      nu_p( pid ) = nu_p_inp;
      for (int j=0; j < 6; j++){
        tangential_disp_ss_x( pid, j ) = 0.;
        tangential_disp_ss_y( pid, j ) = 0.;
        tangential_disp_ss_z( pid, j ) = 0.;
        tangential_disp_idx( pid, j ) = -1;
      }
      total_no_tangential_contacts( pid ) = 0;
    };

  particles->updateParticles( exec_space{}, particles_init_functor );

  // ====================================================
  //                 Wall generation
  // ====================================================
  // Does not set displacements, velocities, etc.
  auto wall = std::make_shared<
    CabanaDEM::Wall<memory_space, DIM>>(exec_space(), 3, output_folder_name);

  // ====================================================
  //            Custom wall initialization
  // ====================================================
  auto x_w = wall->slicePosition();
  auto u_w = wall->sliceVelocity();
  auto normal_w = wall->sliceNormal();
  auto E_w = wall->sliceYoungsMod();
  auto nu_w = wall->slicePoissonsRatio();
  auto G_w = wall->sliceShearMod();

  // For now manually set the wall, use functor later
  // Left, bottom and right walls
  x_w( 0, 0 ) = -2. * radius_p_inp;
  x_w( 0, 1 ) = 0.5;
  x_w( 0, 2 ) = 0.;
  normal_w( 0, 0 ) = 1.;
  normal_w( 0, 1 ) = 0.;
  normal_w( 0, 2 ) = 0.;

  x_w( 1, 0 ) = 0.8;
  x_w( 1, 1 ) = 0.;
  x_w( 1, 2 ) = 0.;
  normal_w( 1, 0 ) = 0.;
  normal_w( 1, 1 ) = 1.;
  normal_w( 1, 2 ) = 0.;

  x_w( 2, 0 ) = 1.4;
  x_w( 2, 1 ) = 0.5;
  x_w( 2, 2 ) = 0.;
  normal_w( 2, 0 ) = -1.;
  normal_w( 2, 1 ) = 0.;
  normal_w( 2, 2 ) = 0.;

  auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
      // Initial conditions: displacements and velocities
      x_w( pid, 0 ) = 0.;
      x_w( pid, 1 ) = -3. * radius_p_inp;
      x_w( pid, 2 ) = 0.;

      u_w( pid, 0 ) = 0.;
      u_w( pid, 1 ) = 0.;
      u_w( pid, 2 ) = 0.;

      normal_w( pid, 0 ) = 0.;
      normal_w( pid, 1 ) = 0.;
      normal_w( pid, 2 ) = 0.;

      E_w( pid ) = E_w_inp;
      G_w( pid ) = G_w_inp;
      nu_w( pid ) = nu_w_inp;
    };
  wall->updateParticles( exec_space{}, init_functor );

  // ====================================================
  //                 Set the neighbours mesh limits
  // ====================================================
  particles->mesh_lo[0] = -4. * radius_p_inp;
  particles->mesh_lo[1] = -4. * radius_p_inp;
  particles->mesh_lo[2] = -4. * radius_p_inp;

  particles->mesh_hi[0] = 1.4 + 4. * radius_p_inp;
  particles->mesh_hi[1] = 1.1 + 4. * radius_p_inp;
  particles->mesh_hi[2] = 4. * radius_p_inp;


  // ====================================================
  //                   Create solver
  // ====================================================
  auto cabana_dem = CabanaDEM::createSolverDEM<memory_space>( inputs, particles, wall,
                                                              force, 3. * radius_p_inp);

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
  // check inputs and write usage
  if ( argc < 2 )
    {
      std::cerr << "Usage: ./01ElasticNormalImpactOfTwoIdenticalParticles  input_file_name output_folder \n";

      std::cerr << "      input_file_name      path to input file name\n";
      std::cerr << "      output_folder        folder to save the data, example 01NormalImpact\n";

      Kokkos::finalize();
      MPI_Finalize();
      return 0;
    }


  GranularDamBreak2D( argv[1], argv[2] );

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
