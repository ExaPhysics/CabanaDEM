#include <fstream>
#include <iostream>
#include <math.h>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaDEM.hpp>

typedef Kokkos::View<double*>   ViewVectorType;
typedef Kokkos::View<double**>  ViewMatrixType;

#define DIM 3
#define MAX_BONDS 27


template <class ExecutionSpace, class ParticleType>
void applyBoundaryConditions( ParticleType& particles )
{
  auto x = particles.slicePosition();
  auto u = particles.sliceVelocity();
  auto au = particles.sliceAcceleration();
  auto force = particles.sliceForce();
  auto torque = particles.sliceTorque();
  auto omega = particles.sliceOmega();
  auto m = particles.sliceMass();
  auto rho = particles.sliceDensity();
  auto rad = particles.sliceRadius();
  auto E = particles.sliceYoungsMod();
  auto nu = particles.slicePoissonsRatio();
  auto G = particles.sliceShearMod();
  auto I = particles.sliceMomentOfInertia();

  auto bond_fn_x = particles.sliceBondFnX();
  auto bond_fn_y = particles.sliceBondFnY();
  auto bond_fn_z = particles.sliceBondFnZ();
  auto bond_ft_x = particles.sliceBondFtX();
  auto bond_ft_y = particles.sliceBondFtY();
  auto bond_ft_z = particles.sliceBondFtZ();
  auto bond_mn_x = particles.sliceBondMnX();
  auto bond_mn_y = particles.sliceBondMnY();
  auto bond_mn_z = particles.sliceBondMnZ();
  auto bond_mt_x = particles.sliceBondMtX();
  auto bond_mt_y = particles.sliceBondMtY();
  auto bond_mt_z = particles.sliceBondMtZ();
  auto bond_idx = particles.sliceBondIdx();
  auto total_no_bonds = particles.sliceTotalNoBonds();

  auto apply_external_full = KOKKOS_LAMBDA( const int i )
    {
        /*
          ====================================
          Add force to the particle i due to contact with particle j
          ====================================
        */
      force( 0, 0 ) = 0.;
      force( 0, 1 ) = 0.;
      force( 0, 2 ) = 0.;
    };

  Kokkos::RangePolicy<ExecutionSpace> policy( 0, u.size() );
  Kokkos::parallel_for( "CabanaDEM::Force::ParticleInfiniteWall", policy,
                        apply_external_full );

}


template <class ExecutionSpace, class ParticleType>
void applyTensileForce( ParticleType& particles, ViewMatrixType ext_frc )
{
  auto x = particles.slicePosition();
  auto u = particles.sliceVelocity();
  auto au = particles.sliceAcceleration();
  auto force = particles.sliceForce();
  auto torque = particles.sliceTorque();
  auto omega = particles.sliceOmega();
  auto m = particles.sliceMass();
  auto rho = particles.sliceDensity();
  auto rad = particles.sliceRadius();
  auto E = particles.sliceYoungsMod();
  auto nu = particles.slicePoissonsRatio();
  auto G = particles.sliceShearMod();
  auto I = particles.sliceMomentOfInertia();

  auto bond_fn_x = particles.sliceBondFnX();
  auto bond_fn_y = particles.sliceBondFnY();
  auto bond_fn_z = particles.sliceBondFnZ();
  auto bond_ft_x = particles.sliceBondFtX();
  auto bond_ft_y = particles.sliceBondFtY();
  auto bond_ft_z = particles.sliceBondFtZ();
  auto bond_mn_x = particles.sliceBondMnX();
  auto bond_mn_y = particles.sliceBondMnY();
  auto bond_mn_z = particles.sliceBondMnZ();
  auto bond_mt_x = particles.sliceBondMtX();
  auto bond_mt_y = particles.sliceBondMtY();
  auto bond_mt_z = particles.sliceBondMtZ();
  auto bond_idx = particles.sliceBondIdx();
  auto total_no_bonds = particles.sliceTotalNoBonds();

  auto apply_external_full = KOKKOS_LAMBDA( const int i )
    {
        /*
          ====================================
          Add force to the particle i due to contact with particle j
          ====================================
        */
      force( i, 0 ) += ext_frc( i, 0 );
      force( i, 1 ) += ext_frc( i, 1 );
      force( i, 2 ) += ext_frc( i, 2 );
    };

  Kokkos::RangePolicy<ExecutionSpace> policy( 0, u.size() );
  Kokkos::parallel_for( "CabanaDEM::Force::ParticleInfiniteWall", policy,
                        apply_external_full );

}

// Simulate two spherical particles colliding head on
void BDEMCantileverBeam3D()
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana Bonded DEM example\n" << std::endl;


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
  // CabanaDEM::Inputs inputs( input_filename );

  // ====================================================
  //                Material parameters
  // ====================================================
  // // Particle material properties
  // double rho_p_inp = inputs["particle_density"];
  // double radius_p_inp = inputs["particle_radius"];
  // double  E_p_inp = inputs["particle_youngs_modulus"];
  // double G_p_inp = inputs["particle_shear_modulus"];
  // double nu_p_inp = inputs["particle_poissons_ratio"];
  // // Wall material properties
  // double  E_w_inp = inputs["wall_youngs_modulus"];
  // double G_w_inp = inputs["wall_shear_modulus"];
  // double nu_w_inp = inputs["wall_poissons_ratio"];
  // // Coefficient of restitution among the interacting bodies
  // double cor_pp_inp = inputs["coefficient_of_restitution_pp"];
  // double cor_pw_inp = inputs["coefficient_of_restitution_pw"];
  // // friction among the interacting bodies
  // double friction_pp_inp = inputs["friction_pp"];
  // double friction_pw_inp = inputs["friction_pw"];
  // Particle material properties
  double rho_p_inp = 2000.;
  double radius_p_inp = 0.5;
  double  E_p_inp = 1e4;
  double G_p_inp = 1e3;
  double nu_p_inp = 0.25;

  // // ====================================================
  // //                Geometric properties
  // // ====================================================
  // double velocity_p_inp = inputs["velocity_p"];
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
  //                 Particle generation
  // ====================================================
  // Does not set displacements, velocities, etc.
  auto particles = CabanaBondedDEM::ParticlesBondedDEM<memory_space, DIM, MAX_BONDS>(exec_space(), 2);
  particles.update_mesh_limits( {-4. * radius_p_inp, -4. * radius_p_inp, -4. * radius_p_inp},
                                {4. * radius_p_inp, 4. * radius_p_inp, 4. * radius_p_inp});

  // ====================================================
  //            Custom particle initialization
  // ====================================================
  // All properties
  auto x = particles.slicePosition();
  auto u = particles.sliceVelocity();
  auto au = particles.sliceAcceleration();
  auto force = particles.sliceForce();
  auto torque = particles.sliceTorque();
  auto omega = particles.sliceOmega();
  auto m = particles.sliceMass();
  auto rho = particles.sliceDensity();
  auto rad = particles.sliceRadius();
  auto E = particles.sliceYoungsMod();
  auto nu = particles.slicePoissonsRatio();
  auto G = particles.sliceShearMod();
  auto I = particles.sliceMomentOfInertia();

  auto bond_fn_x = particles.sliceBondFnX();
  auto bond_fn_y = particles.sliceBondFnY();
  auto bond_fn_z = particles.sliceBondFnZ();
  auto bond_ft_x = particles.sliceBondFtX();
  auto bond_ft_y = particles.sliceBondFtY();
  auto bond_ft_z = particles.sliceBondFtZ();
  auto bond_mn_x = particles.sliceBondMnX();
  auto bond_mn_y = particles.sliceBondMnY();
  auto bond_mn_z = particles.sliceBondMnZ();
  auto bond_mt_x = particles.sliceBondMtX();
  auto bond_mt_y = particles.sliceBondMtY();
  auto bond_mt_z = particles.sliceBondMtZ();
  auto bond_idx = particles.sliceBondIdx();
  auto bond_init_len = particles.sliceBondInitLen();
  auto total_no_bonds = particles.sliceTotalNoBonds();

  auto particles_init_functor = KOKKOS_LAMBDA( const int i )
    {
      double m_p_i = 4. / 3. * M_PI * radius_p_inp * radius_p_inp * radius_p_inp * rho_p_inp;
      double I_p_i = 2. / 5. * m_p_i * radius_p_inp * radius_p_inp;
      x( i, 0 ) = -radius_p_inp + i * (2. * radius_p_inp);
      x( i, 1 ) = 0.;
      x( i, 2 ) = 0.;
      for (int j = 0; j < 3; ++j) {
        u( i, j ) = 0.;
        au( i, j ) = 0.;
        force( i, j ) = 0.;
        torque( i, j ) = 0.;
        omega( i, j ) = 0.;
      }

      m( i ) = m_p_i;
      rho( i ) = rho_p_inp;
      rad( i ) = radius_p_inp;
      E( i ) = E_p_inp;
      nu( i ) = nu_p_inp;
      G( i ) = G_p_inp;
      I( i ) = I_p_i;

      for (int j = 0; j < MAX_BONDS; ++j) {
        bond_fn_x( i, j ) = 0.;
        bond_fn_y( i, j ) = 0.;
        bond_fn_z( i, j ) = 0.;
        bond_ft_x( i, j ) = 0.;
        bond_ft_y( i, j ) = 0.;
        bond_ft_z( i, j ) = 0.;
        bond_mn_x( i, j ) = 0.;
        bond_mn_y( i, j ) = 0.;
        bond_mn_z( i, j ) = 0.;
        bond_mt_x( i, j ) = 0.;
        bond_mt_y( i, j ) = 0.;
        bond_mt_z( i, j ) = 0.;
        bond_idx( i, j ) = -1;
        bond_init_len( i, j ) = 0.;
      }
      total_no_bonds( i ) = 0.;
    };

  particles.updateParticles( exec_space{}, particles_init_functor );
  particles.setupBonds<exec_space>( );

  CabanaBondedDEM::BondedDEMIntegrator<exec_space> integrator(1e-4);

  // ====================================================
  //                   Simulation run
  // ====================================================
  auto dt = 1e-4;
  // auto final_time = 10 * dt;
  auto final_time = 4.;
  // auto final_time = 0.1;
  // auto final_time = 2. * M_PI;
  // auto final_time = 100. * dt;
  auto time = 0.;
  int num_steps = final_time / dt;
  int output_frequency = 100;

  // External force
  double ext_frc_scalar = 1e3;
  ViewMatrixType ext_frc( "ext_frc", x.size(), 3);
  ViewMatrixType::HostMirror host_ext_frc = Kokkos::create_mirror_view( ext_frc );
  // host_ext_frc ( 0, 0 ) = - ext_frc_scalar;
  host_ext_frc ( 0, 0 ) = -ext_frc_scalar;
  host_ext_frc ( 1, 0 ) = ext_frc_scalar;
  Kokkos::deep_copy( ext_frc, host_ext_frc );


  for ( int step = 0; step <= num_steps; step++ )
    {
      integrator.stage1( particles );

      integrator.stage2( particles );

      CabanaBondedDEM::resetForcesAndTorques<exec_space>( particles );
      applyTensileForce<exec_space>( particles, ext_frc );
      CabanaBondedDEM::computeBondedForce<exec_space>( particles, dt );
      applyBoundaryConditions<exec_space>( particles );

      integrator.stage3( particles );

      if ( step % output_frequency == 0 )
        {
          std::cout << "We are at " << step << " " << "/ " << num_steps;
          std::cout << std::endl;
          particles.output( step / output_frequency, step * dt);
        }
    }
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );

  BDEMCantileverBeam3D();

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
