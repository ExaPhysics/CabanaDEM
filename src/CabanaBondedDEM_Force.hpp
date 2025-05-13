#ifndef CabanaBondedDEMForce_HPP
#define CabanaBondedDEMForce_HPP

#include <cmath>

#include <CabanaBondedDEM_Particles.hpp>

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaBondedDEM
{
  template <class ExecutionSpace, class ParticleType>
  void resetForcesAndTorques( ParticleType& particles )
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
    auto bond_init_len = particles.sliceBondInitLen();
    auto total_no_bonds = particles.sliceTotalNoBonds();

    Cabana::deep_copy( force, 0. );
    Cabana::deep_copy( torque, 0. );
  }

  template <class ExecutionSpace, class ParticleType>
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
    auto bond_init_len = particles.sliceBondInitLen();
    auto total_no_bonds = particles.sliceTotalNoBonds();


    auto force_full = KOKKOS_LAMBDA( const int i )
      {
        auto no_bonds_i = total_no_bonds ( i );
        for (int k=0; k < no_bonds_i; k++){
          // Get the bond index
          int j = bond_idx (i, k );
          // if (i == 1) {
          //     std::cout << j;
          //   }
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
          double pos_ij[3] = {x( i, 0 ) - x( j, 0 ),
                              x( i, 1 ) - x( j, 1 ),
                              x( i, 2 ) - x( j, 2 )};

          // squared distance
          double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
          // distance between i and j
          double rij = sqrt(r2ij);

          double g_c = rij - rad ( i ) - rad ( j );
          double a_i = rad ( i ) + g_c / 2.;
          double a_j = rad ( j ) + g_c / 2.;

          const double mass_i = m( i );

          // normal vector passing from j to i
          double normal[3] = {pos_ij[0] / rij, pos_ij[1] / rij, pos_ij[2] / rij};

          double vel_i[3] = {0., 0., 0.};
          double vel_j[3] = {0., 0., 0.};
          double vrel[3] = {0., 0., 0.};
          double vrel_n[3] = {0., 0., 0.};
          double vrel_t[3] = {0., 0., 0.};

          vel_i[0] = u ( i, 0 ) +
            (omega( i, 1 ) * normal[2] - omega( i, 2 ) * normal[1]) * a_i;

          vel_i[1] = u ( i, 1 ) +
            (omega( i, 2 ) * normal[0] - omega( i, 0 ) * normal[2]) * a_i;

          vel_i[2] = u ( i, 2 ) +
            (omega( i, 0 ) * normal[1] - omega( i, 1 ) * normal[0]) * a_i;

          vel_j[0] = u ( j, 0 ) +
            (-omega( j, 1 ) * normal[2] + omega( j, 2 ) * normal[1]) * a_j;

          vel_j[1] = u ( i, 1 ) +
            (-omega( j, 2 ) * normal[0] + omega( j, 0 ) * normal[2]) * a_j;

          vel_j[2] = u ( i, 2 ) +
            (-omega( j, 0 ) * normal[1] + omega( j, 1 ) * normal[0]) * a_j;

          vrel[0] = vel_i[0] - vel_j[0];
          vrel[1] = vel_i[1] - vel_j[1];
          vrel[2] = vel_i[2] - vel_j[2];

          // Calculate normal and tangential components of relative velocity
          double vdp = vrel[0] * normal[0] + vrel[1] * normal[1] + vrel[2] * normal[2];
          vrel_n[0] = vdp * normal[0];
          vrel_n[1] = vdp * normal[1];
          vrel_n[2] = vdp * normal[2];

          vrel_t[0] = vrel[0] - vrel_n[0];
          vrel_t[1] = vrel[1] - vrel_n[1];
          vrel_t[2] = vrel[2] - vrel_n[2];


          double wr[3] = {omega( i, 0 ) - omega( j, 0 ),
                          omega( i, 1 ) - omega( j, 1 ),
                          omega( i, 2 ) - omega( j, 2 )};

          double wdp = wr[0] * normal[0] + wr[1] * normal[1] + wr[2] * normal[2];
          double wn[3] = {wdp*normal[0], wdp*normal[1], wdp*normal[2]};
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
          double rad_sum = rad( i ) + rad( j );
          double radsuminv = 1.0 / (rad_sum);
          // Define the bond constants to be used below
          // This should be input
          double beta_bond = 1e-1;  // input
          double bond_radius_factor = 1.;  // input
          double bond_rad = std::min(rad( i ), rad( j )) * bond_radius_factor;
          double A_bond = M_PI * pow(bond_rad, 2.);
          double I_bond = M_PI * pow(bond_rad, 4.) / 4.0;
          double I_p_bond = I_bond * 2.0;

          // Stiffnesses for Bernoulli beam model and bond damping calculations
          // TODO: These Youngs mod and Shear modulus values should be checked
          double k_n_bond = E ( i ) * A_bond / rad_sum;
          double k_t_bond = G ( i ) * A_bond / rad_sum;
          double k_tor_bond = G ( i ) * I_p_bond / rad_sum;
          double k_ben_bond = E ( i ) * I_bond / rad_sum;

          /*
            ====================================
            ------- Normal bond force -------
            ====================================
          */
          double fn[3] = {0.0, 0.0, 0.0};
          double fn_damped[3] = {0.0, 0.0, 0.0};
          double overlap_n = rij - bond_init_len( i, j );
          fn[0] = -k_n_bond * overlap_n * normal[0];
          fn[1] = -k_n_bond * overlap_n * normal[1];
          fn[2] = -k_n_bond * overlap_n * normal[2];

          double minvel = 1e-5 * std::min( rad ( i ), rad ( j ) ) /dt;
          double dvX = 0.01 * fn[0]*dt;
          double dvY = 0.01 * fn[1]*dt;
          double dvZ = 0.01 * fn[2]*dt;
          double multiplierX = (vrel_n[0] >= 0.0) ? 1.0 : -1.0;
          double multiplierY = (vrel_n[1] >= 0.0) ? 1.0 : -1.0;
          double multiplierZ = (vrel_n[2] >= 0.0) ? 1.0 : -1.0;

          fn_damped[0] = fn[0] + beta_bond*fabs(fn[0]) * multiplierX;
          fn_damped[1] = fn[1] + beta_bond*fabs(fn[1]) * multiplierY;
          fn_damped[2] = fn[2] + beta_bond*fabs(fn[2]) * multiplierZ;
          double beta = - beta_bond*fabs(fn[0]) * multiplierX;
          std::cout << "The beta force is " << beta << std::endl;

          /*
            ====================================
            ------- Tangent bond force -------
            ====================================
          */
          double ft[3] = {bond_ft_x (i, j),
                          bond_ft_y (i, j),
                          bond_ft_z (i, j)};
          double ft_damped[3] = {0.0, 0.0, 0.0};
          double fdp = ft[0] * normal[0] + ft[1] * normal[1] + ft[2] * normal[2];
          double ft_norm[3] = {fdp*normal[0], fdp*normal[1], fdp*normal[2]};
          ft[0] -= ft_norm[0];
          ft[1] -= ft_norm[1];
          ft[2] -= ft_norm[2];

          double tmp = k_t_bond*dt;
          double dft[3] = {-vrel_t[0] * tmp, -vrel_t[1] * tmp, -vrel_t[2] * tmp};

          ft[0] += dft[0];
          ft[1] += dft[1];
          ft[2] += dft[2];

          dvX = 0.01*ft[0]*dt;
          dvY = 0.01*ft[1]*dt;
          dvZ = 0.01*ft[2]*dt;
          multiplierX = (vrel_t[0] >= 0.0) ? 1.0 : -1.0;
          multiplierY = (vrel_t[1] >= 0.0) ? 1.0 : -1.0;
          multiplierZ = (vrel_t[2] >= 0.0) ? 1.0 : -1.0;
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
          double tdp = torq_norm[0] * normal[0] + torq_norm[1] * normal[1] + torq_norm[2] * normal[2];
          torq_norm[0] = tdp * normal[0];
          torq_norm[1] = tdp * normal[1];
          torq_norm[2] = tdp * normal[2];

          tmp = k_tor_bond * dt;
          double dntorque[3] = {-wn[0] * tmp, -wn[1] * tmp, -wn[2] * tmp};
          torq_norm[0] += dntorque[0];
          torq_norm[1] += dntorque[1];
          torq_norm[2] += dntorque[2];

          multiplierX = (wn[0] >= 0.0) ? 1.0 : -1.0;
          multiplierY = (wn[1] >= 0.0) ? 1.0 : -1.0;
          multiplierZ = (wn[2] >= 0.0) ? 1.0 : -1.0;
          ntorq_damped[0] = torq_norm[0] - beta_bond*fabs(torq_norm[0])*multiplierX;
          ntorq_damped[1] = torq_norm[1] - beta_bond*fabs(torq_norm[1])*multiplierY;
          ntorq_damped[2] = torq_norm[2] - beta_bond*fabs(torq_norm[2])*multiplierZ;

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
          tdp = torq_tang[0] * normal[0] + torq_tang[1] * normal[1] + torq_tang[2] * normal[2];
          torq_tang[0] -= tdp * normal[0];
          torq_tang[1] -= tdp * normal[1];
          torq_tang[2] -= tdp * normal[2];

          tmp = k_ben_bond * dt;
          double dttorque[3] = {-wt[0] * tmp, -wt[1] * tmp, -wt[2] * tmp};
          torq_tang[0] += dttorque[0];
          torq_tang[1] += dttorque[1];
          torq_tang[2] += dttorque[2];

          double ttorq_damped[3] = {0.0, 0.0, 0.0};
          multiplierX = (wt[0] >= 0.0) ? 1.0 : -1.0;
          multiplierY = (wt[1] >= 0.0) ? 1.0 : -1.0;
          multiplierZ = (wt[2] >= 0.0) ? 1.0 : -1.0;
          ttorq_damped[0] = torq_tang[0] - beta_bond*fabs(torq_tang[0])*multiplierX;
          ttorq_damped[1] = torq_tang[1] - beta_bond*fabs(torq_tang[1])*multiplierY;
          ttorq_damped[2] = torq_tang[2] - beta_bond*fabs(torq_tang[2])*multiplierZ;

          bond_mt_x (i, j) = ttorq_damped[0];
          bond_mt_y (i, j) = ttorq_damped[1];
          bond_mt_z (i, j) = ttorq_damped[2];

          // Add on cross product term (should not be included in tau_bond_t update)
          double tor[3] = {0.0, 0.0, 0.0};
          // crosspdt(tforce_damped, normal, tor);
          double cri = rij * 1. / rad ( i ) * 1. / ( rad ( i ) + rad ( j ));

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
