#ifndef CabanaBondedDEMParticles_HPP
#define CabanaBondedDEMParticles_HPP

#include <memory>
#include <filesystem> // or #include <filesystem> for C++17 and up

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>


namespace fs = std::filesystem;

#define DEFINE_SLICE_GETTER(NAME, MEMBER, LABEL)                        \
    auto NAME() { return Cabana::slice<0>(MEMBER, LABEL); }       \
    auto NAME() const { return Cabana::slice<0>(MEMBER, LABEL); }


#define DIM 3

namespace CabanaBondedDEM
{
  // =========================
  // Bonded DEM particle array
  // =========================
  template <class MemorySpace, int Dimension, int MaxBonds>
  class ParticlesBondedDEM {
  public:
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;
    static constexpr int dim = Dimension;

    using double_type = Cabana::MemberTypes<double>;
    using int_type = Cabana::MemberTypes<int>;
    using vec_double_type = Cabana::MemberTypes<double[dim]>;
    using vec_int_type = Cabana::MemberTypes<int[dim]>;
    using track_bond_double_type = Cabana::MemberTypes<double[MaxBonds]>;
    using track_bond_int_type = Cabana::MemberTypes<int[MaxBonds]>;

    using aosoa_double_type = Cabana::AoSoA<double_type, memory_space, 1>;
    using aosoa_int_type = Cabana::AoSoA<int_type, memory_space, 1>;
    using aosoa_vec_double_type = Cabana::AoSoA<vec_double_type, memory_space, 1>;
    using aosoa_track_bond_double_type = Cabana::AoSoA<track_bond_double_type, memory_space, 1>;
    using aosoa_track_bond_int_type = Cabana::AoSoA<track_bond_int_type, memory_space, 1>;

    double delta;

    template <class ExecSpace>
    ParticlesBondedDEM(const ExecSpace& exec_space, std::size_t no_of_particles)
      : _no_of_particles(no_of_particles)
    {
      mesh_lo = {0., 0., 0.};
      mesh_hi = {0., 0., 0.};
      delta = 0.;
      resize(_no_of_particles);
      createParticles(exec_space);
    }

    void update_mesh_limits(const std::array<double, 3>& lo,
                            const std::array<double, 3>& hi) {
      mesh_lo = lo;
      mesh_hi = hi;
    }

    template <class ExecSpace>
    void createParticles(const ExecSpace& exec_space) {
      auto x = slicePosition();
      Kokkos::parallel_for("create_particles", Kokkos::RangePolicy<ExecSpace>(0, x.size()),
                           KOKKOS_LAMBDA(const int i) {
                             for (int j = 0; j < dim; j++) {
                               // x(i, j) = some initial value;
                             }
                           });
    }

    template <class ExecSpace, class FunctorType>
    void updateParticles(const ExecSpace& exec_space, const FunctorType& functor) {
      Kokkos::parallel_for("update_particles", Kokkos::RangePolicy<ExecSpace>(0, _no_of_particles),
                           KOKKOS_LAMBDA(const int pid) { functor(pid); });
    }

    template <class ExecSpace>
    void setupBonds() {
      // All properties
      auto x = slicePosition();
      auto u = sliceVelocity();
      auto au = sliceAcceleration();
      auto force = sliceForce();
      auto torque = sliceTorque();
      auto omega = sliceOmega();
      auto m = sliceMass();
      auto rho = sliceDensity();
      auto rad = sliceRadius();
      auto E = sliceYoungsMod();
      auto nu = slicePoissonsRatio();
      auto G = sliceShearMod();
      auto I = sliceMomentOfInertia();

      auto bond_fn_x = sliceBondFnX();
      auto bond_fn_y = sliceBondFnY();
      auto bond_fn_z = sliceBondFnZ();
      auto bond_ft_x = sliceBondFtX();
      auto bond_ft_y = sliceBondFtY();
      auto bond_ft_z = sliceBondFtZ();
      auto bond_mn_x = sliceBondMnX();
      auto bond_mn_y = sliceBondMnY();
      auto bond_mn_z = sliceBondMnZ();
      auto bond_mt_x = sliceBondMtX();
      auto bond_mt_y = sliceBondMtY();
      auto bond_mt_z = sliceBondMtZ();
      auto bond_idx = sliceBondIdx();
      auto bond_init_len = sliceBondInitLen();
      auto total_no_bonds = sliceTotalNoBonds();

      auto setup_bonds_functor = KOKKOS_LAMBDA( const int i, const int j )
        {
          double pos_ij[3] = {x( i, 0 ) - x( j, 0 ),
                              x( i, 1 ) - x( j, 1 ),
                              x( i, 2 ) - x( j, 2 )};
          // squared distance
          double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
          // distance between i and j
          double rij = sqrt(r2ij);

          // Since all the particles have the same radius. This will change for
          // inhomogeneous radius distribution
          auto bond_cond = 3. * rad ( i );

          if (rij < bond_cond){
            bond_idx( i, total_no_bonds( i ) ) = j;
            bond_init_len( i, total_no_bonds( i ) ) = rij;
            total_no_bonds( i ) += 1;
          }
        };

      // ==========================
      // Create the neighbours
      // ==========================
      using neighbor_type =
        Cabana::VerletList<memory_space, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;

      double mesh_min[3] = {
        mesh_lo[0],
        mesh_lo[1],
        mesh_lo[2]};
      double mesh_max[3] = {
        mesh_hi[0],
        mesh_hi[1],
        mesh_hi[2]};
      neighbor_type neighbors;
      neighbors = neighbor_type( x, 0, x.size(),
                                 4. * rad ( 0 ), 2.,
                                 mesh_min, mesh_max );
      // ==========================
      // Create the neighbours ends
      // ==========================

      Kokkos::RangePolicy<ExecSpace> policy(0, x.size());

      Cabana::neighbor_parallel_for( policy,
                                     setup_bonds_functor,
                                     neighbors,
                                     Cabana::FirstNeighborsTag(),
                                     Cabana::SerialOpTag(),
                                     "CabanaBondedDEM::setupBonds" );
      Kokkos::fence();
    }

    DEFINE_SLICE_GETTER(slicePosition, _x, "x")
    DEFINE_SLICE_GETTER(sliceVelocity, _u, "u")
    DEFINE_SLICE_GETTER(sliceAcceleration, _au, "au")
    DEFINE_SLICE_GETTER(sliceForce, _force, "force")
    DEFINE_SLICE_GETTER(sliceTorque, _torque, "torque")
    DEFINE_SLICE_GETTER(sliceOmega, _omega, "omega")
    DEFINE_SLICE_GETTER(sliceMass, _m, "m")
    DEFINE_SLICE_GETTER(sliceDensity, _rho, "rho")
    DEFINE_SLICE_GETTER(sliceRadius, _rad, "rad")
    DEFINE_SLICE_GETTER(sliceYoungsMod, _E, "E")
    DEFINE_SLICE_GETTER(slicePoissonsRatio, _nu, "nu")
    DEFINE_SLICE_GETTER(sliceShearMod, _G, "G")
    DEFINE_SLICE_GETTER(sliceMomentOfInertia, _I, "I")

    DEFINE_SLICE_GETTER(sliceBondFnX, _bond_fn_x, "bond_fn_x")
    DEFINE_SLICE_GETTER(sliceBondFnY, _bond_fn_y, "bond_fn_y")
    DEFINE_SLICE_GETTER(sliceBondFnZ, _bond_fn_z, "bond_fn_z")
    DEFINE_SLICE_GETTER(sliceBondFtX, _bond_ft_x, "bond_ft_x")
    DEFINE_SLICE_GETTER(sliceBondFtY, _bond_ft_y, "bond_ft_y")
    DEFINE_SLICE_GETTER(sliceBondFtZ, _bond_ft_z, "bond_ft_z")
    DEFINE_SLICE_GETTER(sliceBondMnX, _bond_mn_x, "bond_mn_x")
    DEFINE_SLICE_GETTER(sliceBondMnY, _bond_mn_y, "bond_mn_y")
    DEFINE_SLICE_GETTER(sliceBondMnZ, _bond_mn_z, "bond_mn_z")
    DEFINE_SLICE_GETTER(sliceBondMtX, _bond_mt_x, "bond_mt_x")
    DEFINE_SLICE_GETTER(sliceBondMtY, _bond_mt_y, "bond_mt_y")
    DEFINE_SLICE_GETTER(sliceBondMtZ, _bond_mt_z, "bond_mt_z")
    DEFINE_SLICE_GETTER(sliceBondIdx, _bond_idx, "bond_idx")
    DEFINE_SLICE_GETTER(sliceBondInitLen, _bond_init_len, "bond_init_len")
    DEFINE_SLICE_GETTER(sliceTotalNoBonds, _total_no_bonds, "total_no_bonds")

    void resize(std::size_t n) {
      _x.resize(n);
      _u.resize(n);
      _au.resize(n);
      _force.resize(n);
      _torque.resize(n);
      _omega.resize(n);
      _m.resize(n);
      _rho.resize(n);
      _rad.resize(n);
      _E.resize(n);
      _nu.resize(n);
      _G.resize(n);
      _I.resize(n);

      _bond_fn_x.resize(n);
      _bond_fn_y.resize(n);
      _bond_fn_z.resize(n);
      _bond_ft_x.resize(n);
      _bond_ft_y.resize(n);
      _bond_ft_z.resize(n);
      _bond_mn_x.resize(n);
      _bond_mn_y.resize(n);
      _bond_mn_z.resize(n);
      _bond_mt_x.resize(n);
      _bond_mt_y.resize(n);
      _bond_mt_z.resize(n);
      _bond_idx.resize(n);
      _bond_init_len.resize(n);
      _total_no_bonds.resize(n);
    }

    void output(int step, double time, bool use_reference = true) {
#ifdef Cabana_ENABLE_HDF5
      Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
                                                              h5_config,
                                                              "particles",
                                                              MPI_COMM_WORLD,
                                                              step,
                                                              time,
                                                              _no_of_particles,
                                                              slicePosition(),
                                                              sliceVelocity(),
                                                              sliceOmega(),
                                                              sliceAcceleration(),
                                                              sliceForce(),
                                                              sliceMass(),
                                                              sliceDensity(),
                                                              sliceRadius(),

                                                              sliceBondFnX(),
                                                              sliceBondFnY(),
                                                              sliceBondFnZ(),
                                                              sliceBondFtX(),
                                                              sliceBondFtY(),
                                                              sliceBondFtZ(),
                                                              sliceBondMnX(),
                                                              sliceBondMnY(),
                                                              sliceBondMnZ(),
                                                              sliceBondMtX(),
                                                              sliceBondMtY(),
                                                              sliceBondMtZ(),
                                                              sliceBondIdx(),
                                                              sliceBondInitLen(),
                                                              sliceTotalNoBonds());
#else
      std::cout << "No particle output enabled." << std::endl;
#endif
    }

  private:
    std::size_t _no_of_particles;

    // Particle data
    aosoa_vec_double_type _x, _u, _au, _force, _torque, _omega;
    aosoa_double_type _m, _rho, _rad, _E, _nu, _G, _I;
    aosoa_track_bond_double_type _bond_fn_x, _bond_fn_y, _bond_fn_z;
    aosoa_track_bond_double_type _bond_ft_x, _bond_ft_y, _bond_ft_z;
    aosoa_track_bond_double_type _bond_mn_x, _bond_mn_y, _bond_mn_z;
    aosoa_track_bond_double_type _bond_mt_x, _bond_mt_y, _bond_mt_z;
    aosoa_track_bond_double_type _bond_init_len;
    aosoa_track_bond_int_type _bond_idx;
    aosoa_int_type _total_no_bonds;

    std::array<double, dim> mesh_lo{};
    std::array<double, dim> mesh_hi{};

#ifdef Cabana_ENABLE_HDF5
    Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
#endif
  };

} // namespace DEM

#undef DEFINE_SLICE_GETTER

#endif


// All properties
// auto x = particles.slicePosition();
// auto u = particles.sliceVelocity();
// auto au = particles.sliceAcceleration();
// auto force = particles.sliceForce();
// auto torque = particles.sliceTorque();
// auto omega = particles.sliceOmega();
// auto m = particles.sliceMass();
// auto rho = particles.sliceDensity();
// auto rad = particles.sliceRadius();
// auto E = particles.sliceYoungsMod();
// auto nu = particles.slicePoissonsRatio();
// auto G = particles.sliceShearMod();
// auto I = particles.sliceMomentOfInertia();

// auto bond_fn_x = particles.sliceBondFnX();
// auto bond_fn_y = particles.sliceBondFnY();
// auto bond_fn_z = particles.sliceBondFnZ();
// auto bond_ft_x = particles.sliceBondFtX();
// auto bond_ft_y = particles.sliceBondFtY();
// auto bond_ft_z = particles.sliceBondFtZ();
// auto bond_mn_x = particles.sliceBondMnX();
// auto bond_mn_y = particles.sliceBondMnY();
// auto bond_mn_z = particles.sliceBondMnZ();
// auto bond_mt_x = particles.sliceBondMtX();
// auto bond_mt_y = particles.sliceBondMtY();
// auto bond_mt_z = particles.sliceBondMtZ();
// auto bond_idx = particles.sliceBondIdx();
// auto bond_init_len = particles.sliceBondInitLen();
// auto total_no_bonds = particles.sliceTotalNoBonds();
