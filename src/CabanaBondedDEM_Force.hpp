#ifndef CABANAForce_HPP
#define CABANAForce_HPP

#include <cmath>

#include <CabanaBondedDEM_Particles.hpp>

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaBondedDEM
{
  template <class ParticleType>
  void computeBondedForce( ParticleType& particles, double dt )
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
    auto total_no_bonds = particles.totalNoBonds();



    auto force_full = KOKKOS_LAMBDA( const int i )
      {
        auto no_bonds_i = total_no_bonds ( i );
        for (int j=0; j < no_bonds; j++){
          /*
            Common to all equations in SPH.

            We compute:
            1.the vector passing from j to i
            2. Distance between the points i and j
            3. Distance square between the points i and j
            4. Velocity vector difference between i and j
            5. Kernel value
            6. Derivative of kernel value
          */
          double pos_i[3] = {x( i, 0 ),
                             x( i, 1 ),
                             x( i, 2 )};

          double pos_j[3] = {x( j, 0 ),
                             x( j, 1 ),
                             x( j, 2 )};

          double pos_ij[3] = {x( i, 0 ) - x( j, 0 ),
                              x( i, 1 ) - x( j, 1 ),
                              x( i, 2 ) - x( j, 2 )};

          // squared distance
          double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
          // distance between i and j
          double rij = sqrt(r2ij);

          const double mass_i = m( i );

          // normal vector passing from j to i
          double nij_x = pos_ij[0] / rij;
          double nij_y = pos_ij[1] / rij;
          double nij_z = pos_ij[2] / rij;

          double vel_i[3] = {0., 0., 0.};

          vel_i[0] = u ( i, 0 ) +
            (omega( i, 1 ) * nij_z - omega( i, 2 ) * nij_y) * a_i;

          vel_i[1] = u ( i, 1 ) +
            (omega( i, 2 ) * nij_x - omega( i, 0 ) * nij_z) * a_i;

          vel_i[2] = u ( i, 2 ) +
            (omega( i, 0 ) * nij_y - omega( i, 1 ) * nij_x) * a_i;

          double vel_j[3] = {0., 0., 0.};

          vel_j[0] = u ( j, 0 ) +
            (-omega( j, 1 ) * nij_z + omega( j, 2 ) * nij_y) * a_j;

          vel_j[1] = u ( i, 1 ) +
            (-omega( j, 2 ) * nij_x + omega( j, 0 ) * nij_z) * a_j;

          vel_j[2] = u ( i, 2 ) +
            (-omega( j, 0 ) * nij_y + omega( j, 1 ) * nij_x) * a_j;

          // Now the relative velocity of particle i w.r.t j at the contact
          // point is
          double vel_ij[3] = {vel_i[0] - vel_j[0],
                              vel_i[1] - vel_j[1],
                              vel_i[2] - vel_j[2]};

          // normal velocity magnitude
          double vij_dot_nij = vel_ij[0] * nij_x + vel_ij[1] * nij_y + vel_ij[2] * nij_z;
          double vn_x = vij_dot_nij * nij_x;
          double vn_y = vij_dot_nij * nij_y;
          double vn_z = vij_dot_nij * nij_z;

          // tangential velocity
          double vt_x = vel_ij[0] - vn_x;
          double vt_y = vel_ij[1] - vn_y;
          double vt_z = vel_ij[2] - vn_z;

          double wr[3] = {omega( i, 0 ) - omega( j, 0 ),
                          omega( i, 1 ) - omega( j, 1 ),
                          omega( i, 2 ) - omega( j, 2 )};

          double wdp = wr[0] * nij_x + wr[1] * nij_y + wr[2] * nij_z;
          double wn[3] = {wdp*nij_x, wdp*nij_y, wdp*nij_z};
          double wt[3] = {wr[0] - wn[0], wr[1] - wn[1], wr[2] - wn[2]};

          /*
            ====================================
            End: common to all equations in SPH.
            ====================================
          */

          /*
            ====================================
            // So we are computing force on particle i, base your equations on this point
            ====================================
          */

          /*
            ====================================
            ------- Normal bond force -------
            ====================================
          */
          double fn[0] = {0.0, 0.0, 0.0};
          double fn_damped[0] = {0.0, 0.0, 0.0};
          double k_n_bond = 1.;
          fn[0] = -k_n_bond * overlap_n * nij_x;
          fn[1] = -k_n_bond * overlap_n * nij_y;
          fn[2] = -k_n_bond * overlap_n * nij_z;

          double minvel = 1e-5 * std::min(rad1,rad2) /dt;
          double dvX = 0.01 * fn[0]*dt;
          double dvY = 0.01 * fn[1]*dt;
          double dvZ = 0.01 * fn[2]*dt;
          double multiplierX = (vrel_n[0] >= 0.0) ? 1.0 : -1.0;
          double multiplierY = (vrel_n[1] >= 0.0) ? 1.0 : -1.0;
          double multiplierZ = (vrel_n[2] >= 0.0) ? 1.0 : -1.0;

          double beta_bond = 1.;
          fn_damped[0] = fn[0] - beta_bond*fabs(fn[0]) * multiplierX;
          fn_damped[1] = fn[1] - beta_bond*fabs(fn[1]) * multiplierY;
          fn_damped[2] = fn[2] - beta_bond*fabs(fn[2]) * multiplierZ;

          /*
            ====================================
            ------- Tangent bond force -------
            ====================================
          */
          double ft[3] = {bond_ft_x (i, j),
                          bond_ft_y (i, j),
                          bond_ft_z (i, j)};
          double ft_damped[3] = {0.0, 0.0, 0.0};
          double fdp = ft[0] * nij_x + ft[1] * nij_y + ft[2] * nij_z;
          double ft_norm[3] = {fdp*nij_x, fdp*nij_y, fdp*nij_z};
          ft[0] -= ft_norm[0];
          ft[1] -= ft_norm[1];
          ft[2] -= ft_norm[2];

          double tmp = k_t_bond*dt;
          double dft[3] = {-vrel_t[0] * tmp, -vrel_t[0] * tmp, -vrel_t[0] * tmp};

          ft[0] += dft[0];
          ft[1] += dft[1];
          ft[2] += dft[2];

          dvX = 0.01*ft[0]*dt;
          dvY = 0.01*ft[1]*dt;
          dvZ = 0.01*ft[2]*dt;
          multiplierX = (vrel_t[0] >= 0.0) ? 1.0 : -1.0;
          multiplierY = (vrel_t[1] >= 0.0) ? 1.0 : -1.0;
          multiplierZ = (vrel_t[2] >= 0.0) ? 1.0 : -1.0;
          double k_n_bond = 1.;
          ft_damped[0] = ft[0] - beta_bond*fabs(ft[0])*multiplierX;
          ft_damped[1] = ft[1] - beta_bond*fabs(ft[1])*multiplierY;
          ft_damped[2] = ft[2] - beta_bond*fabs(ft[2])*multiplierZ;

          bond_ft_x (i, j) = ft_damped[0];
          bond_ft_y (i, j) = ft_damped[1];
          bond_ft_z (i, j) = ft_damped[2];

          /*
            ====================================
            ------- Normal bond torque -------
            ====================================
          */
          double torq_norm[3] = {bond_mn_x (i, j),
                               bond_mn_y (i, j),
                               bond_mn_z (i, j)};
          double ntorq_damped[3] = {0.0, 0.0, 0.0};
          double tdp = torq_norm[0] * nij_x + torq_norm[1] * nij_y + torq_norm[2] * nij_z;
          torq_norm[0] = tdp * nij_x;
          torq_norm[1] = tdp * nij_y;
          torq_norm[2] = tdp * nij_z;

          tmp = k_tor_bond * dt;
          double dntorque[3] = {-wn[0] * tmp, -wn[1] * tmp, -wn[2] * tmp};
          torq_norm[0] += dntorque[0];
          torq_norm[1] += dntorque[1];
          torq_norm[2] += dntorque[2];
          ntorq_damped[0] = torq_norm[0] - DEM::beta_bond*fabs(torq_norm[0])*amrex::Math::copysign(1.0,wn[0]);
          ntorq_damped[1] = torq_norm[1] - DEM::beta_bond*fabs(torq_norm[1])*amrex::Math::copysign(1.0,wn[1]);
          ntorq_damped[2] = torq_norm[2] - DEM::beta_bond*fabs(torq_norm[2])*amrex::Math::copysign(1.0,wn[2]);

          bond_mn_x (i, j)= ntorq_damped[0];
          bond_mn_y (i, j)= ntorq_damped[1];
          bond_mn_z (i, j)= ntorq_damped[2];

          /*
            ====================================
            ------- Tangential bond torque -----
            ====================================
          */
          double torq_tang[3] = {bond_mt_x (i, j),
                                 bond_mt_y (i, j),
                                 bond_mt_z (i, j)};
          double ttorq_damped[3] = {0.0, 0.0, 0.0};
          tdp = torq_tang[0] * nij_x + torq_tang[1] * nij_y + torq_tang[2] * nij_z;
          torq_tang[0] -= tdp * nij_x;
          torq_tang[1] -= tdp * nij_y;
          torq_tang[2] -= tdp * nij_z;

          tmp = k_ben_bond * dt;
          double dttorque[3] = {-wt[0] * tmp, -wt[1] * tmp, -wt[2] * tmp};
          torq_tang[0] += dttorque[0];
          torq_tang[1] += dttorque[1];
          torq_tang[2] += dttorque[2];
          ttorq_damped[0] = torq_tang[0] - DEM::beta_bond*fabs(torq_tang[0])*amrex::Math::copysign(1.0,wt[0]);
          ttorq_damped[1] = torq_tang[1] - DEM::beta_bond*fabs(torq_tang[1])*amrex::Math::copysign(1.0,wt[1]);
          ttorq_damped[2] = torq_tang[2] - DEM::beta_bond*fabs(torq_tang[2])*amrex::Math::copysign(1.0,wt[2]);

          bond_mt_x (i, j)= ttorq_damped[0];
          bond_mt_y (i, j)= ttorq_damped[1];
          bond_mt_z (i, j)= ttorq_damped[2];

          // Add on cross product term (should not be included in tau_bond_t update)
          double tor[THREEDIM] = {0.0, 0.0, 0.0};
          crosspdt(tforce_damped, normal, tor);
          double cri = distmag * rad1 * radsuminv;

          /*
            ====================================
            Add force to the particle i due to contact with particle j
            ====================================
          */
          force( i, 0 ) += fn_damped[0] + ft_damped[0];
          force( i, 1 ) += fn_damped[1] + ft_damped[1];
          force( i, 2 ) += fn_damped[2] + ft_damped[2];

          torque( i, 0 ) += ntorq_damped[0] + ttorq_damped[0] - cri * tor[0];
          torque( i, 1 ) += ntorq_damped[1] + ttorq_damped[1] - cri * tor[1];
          torque( i, 2 ) += ntorq_damped[2] + ttorq_damped[2] - cri * tor[2];
        }
      };

    Kokkos::RangePolicy<ExecutionSpace> policy( 0, u.size() );
    Kokkos::parallel_for( "CabanaDEM::Force::ParticleInfiniteWall", policy,
                          force_full );

  }
}

#endif
