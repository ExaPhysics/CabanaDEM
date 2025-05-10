#ifndef CabanaBondedDEMParticles_HPP
#define CabanaBondedDEMParticles_HPP

#include <memory>
#include <filesystem> // or #include <filesystem> for C++17 and up

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>


namespace fs = std::filesystem;

#define DEFINE_SLICE_GETTER(NAME, MEMBER)                         \
    auto NAME() { return Cabana::slice<0>(MEMBER, #NAME); }       \
    auto NAME() const { return Cabana::slice<0>(MEMBER, #NAME); }


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

    std::array<double, dim> mesh_lo{};
    std::array<double, dim> mesh_hi{};

    template <class ExecSpace>
    ParticlesBondedDEM(const ExecSpace& exec_space, std::size_t no_of_particles, std::string output_folder_name)
      : _no_of_particles(no_of_particles), _output_folder_name(std::move(output_folder_name))
    {
      if (!fs::exists(_output_folder_name))
        fs::create_directory(_output_folder_name);

      resize(_no_of_particles);
      createParticles(exec_space);
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

    DEFINE_SLICE_GETTER(slicePosition, _x)
    DEFINE_SLICE_GETTER(sliceVelocity, _u)
    DEFINE_SLICE_GETTER(sliceAcceleration, _au)
    DEFINE_SLICE_GETTER(sliceForce, _force)
    DEFINE_SLICE_GETTER(sliceTorque, _torque)
    DEFINE_SLICE_GETTER(sliceOmega, _omega)
    DEFINE_SLICE_GETTER(sliceMass, _m)
    DEFINE_SLICE_GETTER(sliceDensity, _rho)
    DEFINE_SLICE_GETTER(sliceRadius, _rad)
    DEFINE_SLICE_GETTER(sliceYoungsMod, _E)
    DEFINE_SLICE_GETTER(slicePoissonsRatio, _nu)
    DEFINE_SLICE_GETTER(sliceShearMod, _G)
    DEFINE_SLICE_GETTER(sliceMomentOfInertia, _I)

    DEFINE_SLICE_GETTER(sliceBondFnX, _bond_fn_x)
    DEFINE_SLICE_GETTER(sliceBondFnY, _bond_fn_y)
    DEFINE_SLICE_GETTER(sliceBondFnZ, _bond_fn_z)
    DEFINE_SLICE_GETTER(sliceBondFtX, _bond_ft_x)
    DEFINE_SLICE_GETTER(sliceBondFtY, _bond_ft_y)
    DEFINE_SLICE_GETTER(sliceBondFtZ, _bond_ft_z)
    DEFINE_SLICE_GETTER(sliceBondMnX, _bond_mn_x)
    DEFINE_SLICE_GETTER(sliceBondMnY, _bond_mn_y)
    DEFINE_SLICE_GETTER(sliceBondMnZ, _bond_mn_z)
    DEFINE_SLICE_GETTER(sliceBondMtX, _bond_mt_x)
    DEFINE_SLICE_GETTER(sliceBondMtY, _bond_mt_y)
    DEFINE_SLICE_GETTER(sliceBondMtZ, _bond_mt_z)
    DEFINE_SLICE_GETTER(sliceBondIdx, _bond_idx)
    DEFINE_SLICE_GETTER(sliceTotalNoBonds, _total_no_bonds)

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
      _total_no_bonds.resize(n);
    }

    void output(int step, double time, bool use_reference = true) {
#ifdef Cabana_ENABLE_HDF5
      Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
                                                              h5_config,
                                                              _output_folder_name + "/particles",
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
                                                              sliceTotalNoBonds());
#else
      std::cout << "No particle output enabled." << std::endl;
#endif
    }

  private:
    std::size_t _no_of_particles;
    std::string _output_folder_name;

    // Particle data
    aosoa_vec_double_type _x, _u, _au, _force, _torque, _omega;
    aosoa_double_type _m, _rho, _rad, _E, _nu, _G, _I;
    aosoa_track_bond_double_type _bond_fn_x, _bond_fn_y, _bond_fn_z;
    aosoa_track_bond_double_type _bond_ft_x, _bond_ft_y, _bond_ft_z;
    aosoa_track_bond_double_type _bond_mn_x, _bond_mn_y, _bond_mn_z;
    aosoa_track_bond_double_type _bond_mt_x, _bond_mt_y, _bond_mt_z;
    aosoa_track_bond_int_type _bond_idx;
    aosoa_int_type _total_no_bonds;

#ifdef Cabana_ENABLE_HDF5
    Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
#endif
  };

} // namespace DEM

#undef DEFINE_SLICE_GETTER

#endif
