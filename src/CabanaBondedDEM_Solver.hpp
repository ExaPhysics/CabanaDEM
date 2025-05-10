#ifndef CABANABondedSolver_HPP
#define CABANABondedSolver_HPP

#include <CabanaBondedDEM_Particles.hpp>
#include <CabanaBondedDEM_Force.hpp>
#include <CabanaBondedDEM_Integrator.hpp>

namespace CabanaBondedDEM
{
  // template <class MemorySpace, class ParticleType>
  // class SolverBondedDEM
  // {
  // public:
  //   using memory_space = MemorySpace;
  //   using exec_space = typename memory_space::execution_space;

  //   using particle_type = ParticleType;
  //   using integrator_type = BondedDEMIntegrator<exec_space>;

  //   // TODO, check this with odler examples
  //   using neighbor_type =
  //     Cabana::VerletList<memory_space, Cabana::FullNeighborTag,
  //                        Cabana::VerletLayout2D, Cabana::TeamOpTag>;
  //   using neigh_iter_tag = Cabana::SerialOpTag;

  //   SolverBondedDEM(std::shared_ptr<particle_type> _particles,
  //                   double _delta)
  //     : particles( _particles )
  //   {
  //     num_steps = inputs["num_steps"];
  //     output_frequency = inputs["output_frequency"];

  //     // Create integrator.
  //     dt = inputs["timestep"];
  //     integrator = std::make_shared<integrator_type>( dt );

  //     double mesh_min[3] = {
  //       particles->mesh_lo[0],
  //       particles->mesh_lo[1],
  //             particles->mesh_lo[2]};
  //     double mesh_max[3] = {
  //       particles->mesh_hi[0],
  //       particles->mesh_hi[1],
  //       particles->mesh_hi[2]};
  //     auto x = particles->slicePosition();
  //     // This will be changed (No hard coded values)
  //     auto cell_ratio = 1.0;
  //     neighbors = std::make_shared<neighbor_type>( x, 0, x.size(),
  //                                                  delta, cell_ratio,
  //                                                  mesh_min, mesh_max );
  //   }

  //   void run()
  //   {
  //     auto x = particles->slicePosition();
  //     auto cell_ratio = 1.0;
  //     double mesh_min[3] = {
  //       particles->mesh_lo[0],
  //       particles->mesh_lo[1],
  //       particles->mesh_lo[2]};
  //     double mesh_max[3] = {
  //       particles->mesh_hi[0],
  //       particles->mesh_hi[1],
  //       particles->mesh_hi[2]};
  //     // Main timestep loop.
  //     for ( int step = 0; step <= num_steps; step++ )
  //       {
  //         integrator->stage1( *particles );

  //         integrator->stage2( *particles );


  //         // // update the neighbours
  //         // neighbors->build( x, 0, x.size(), delta,
  //         //                   cell_ratio, mesh_min, mesh_max );
  //         // Compute the interaction force
  //         CabanaBondedDEM::computeBondedForce( *particles, dt );


  //         integrator->stage3( *particles );

  //         output( step );
  //       }
  //   }

  //   void output( const int step )
  //   {
  //     if ( step % output_frequency == 0 )
  //       {
  //         std::cout << "We are at " << step << " " << "/ " << num_steps;
  //         std::cout << std::endl;
  //         particles->output( step / output_frequency, step * dt);
  //       }
  //   }

  //   int num_steps;
  //   int output_frequency;
  //   double dt;

  // protected:
  //   std::shared_ptr<particle_type> particles;
  //   std::shared_ptr<integrator_type> integrator;
  //   double delta;
  // };


  // //---------------------------------------------------------------------------//
  // // Creation method.
  // template <class MemorySpace, class ParticleType, class WallType,
  //           class ForceModel>
  // auto createSolverBondedDEM(std::shared_ptr<ParticleType> particles,
  //                            double delta)
  // {
  //   return std::make_shared<
  //     SolverBondedDEM<MemorySpace, ParticleType>>(particles, delta);
  // }

}

#endif
