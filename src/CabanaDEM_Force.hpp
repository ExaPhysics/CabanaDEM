#ifndef CABANAForce_HPP
#define CABANAForce_HPP

#include <cmath>

#include <CabanaDEM_Particles.hpp>

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaDEM
{
  template <class ExecutionSpace>
  class Force
  {
    using exec_space = ExecutionSpace;

  public:
    double cor_pp, cor_pw, friction_pp, friction_pw;

    Force(double cor_pp, double cor_pw, double friction_pp,
          double friction_pw):
      cor_pp (cor_pp),
      cor_pw (cor_pw),
      friction_pp (friction_pp),
      friction_pw (friction_pw)

    {
      // cor_pp = cor_pp;
      // cor_pw = cor_pw;
      // friction_pp = friction_pp;
      // friction_pw = friction_pw;
      }

      ~Force() {}

    template <class ParticleType>
    void makeForceTorqueZeroOnParticle(ParticleType& particles)
    {
      auto force = particles.sliceForce();
      auto torque = particles.sliceTorque();

      Cabana::deep_copy( force, 0. );
      Cabana::deep_copy( torque, 0. );
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void computeForceFullParticleParticle(ParticleType& particles,
                                          const NeighListType& neigh_list,
                                          ParallelType& neigh_op_tag,
                                          double dt)
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
      auto tangential_disp_ss_x = particles.sliceTangentialDispSSx();
      auto tangential_disp_ss_y = particles.sliceTangentialDispSSy();
      auto tangential_disp_ss_z = particles.sliceTangentialDispSSz();
      auto tangential_disp_idx = particles.sliceTangentialDispIdx();
      auto total_no_tangential_contacts = particles.sliceTotalNoTangentialContacts();

      auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
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

        // const double mass_i = aosoa_mass( i );
        const double mass_j = m ( j );

        // Find the overlap amount
        double overlap =  rad ( i ) + rad ( j ) - rij;

        double a_i = rad ( i ) - overlap / 2.;
        double a_j = rad ( j ) - overlap / 2.;

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

        vel_j[1] = u ( j, 1 ) +
        (-omega( j, 2 ) * nij_x + omega( j, 0 ) * nij_z) * a_j;

        vel_j[2] = u ( j, 2 ) +
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

        /*
          ====================================
          End: common to all equations in SPH.
          ====================================
        */
        // find the force if the particles are overlapping
        if (overlap > 0.) {
          // Check if the particle is being tracked
          // if the particle is being tracked then update the
          // tracked tangential index
          auto found = 0;
          auto found_at = -1;
          for ( std::size_t k = 0; k < total_no_tangential_contacts( i ); ++k )
            {
              if (j == tangential_disp_idx( i, k )) {
                found = 1;
                found_at = k;
              }
            }

          if (found == 1)
            {
              // don't do anything
            }
          else{
            // implies this is a new contact
            // so add it to the contacting indices and increase the total
            // count of the contacting indices

            // create a new index at the end of the tracking
            total_no_tangential_contacts ( i ) += 1;
            found_at = total_no_tangential_contacts ( i ) - 1;
            tangential_disp_idx ( i, found_at ) = j;
            tangential_disp_ss_x ( i, found_at ) = 0.;
            tangential_disp_ss_y ( i, found_at ) = 0.;
            tangential_disp_ss_z ( i, found_at ) = 0.;
          }

          // Compute stiffness
          // effective Young's modulus
          double tmp_1 = (1. - (nu( i )*nu( i ))) / E( i );
          double tmp_2 = (1. - (nu( j )*nu( j ))) / E( j );
          double E_eff = 1. / (tmp_1 + tmp_2);
          double tmp_3 = 1. / rad( i );
          double tmp_4 = 1. / rad( j );
          double R_eff = 1. / (tmp_3 + tmp_4);
          // # Eq 4 [1]
          double kn = 4. / 3. * E_eff * sqrt(R_eff);

          // # compute damping coefficient
          tmp_1 = log(cor_pp);
          // std::cout << "log value is" << tmp_1 << std::endl;
          // tmp_1 = 0.;
          tmp_2 = tmp_1 * tmp_1 + M_PI * M_PI;
          double beta = tmp_1 / sqrt(tmp_2);
          double S_n = 2. * E_eff * sqrt(R_eff * overlap);
          tmp_1 = 1. / m( i );
          tmp_2 = 1. / m( j );
          double m_eff = 1. / (tmp_1 + tmp_2);
          double eta_n = -2. * sqrt(5./6.) * beta * sqrt(S_n * m_eff);

          // # Compute the tangential stiffness
          tmp_1 = (2. - nu( i )) / G( i );
          tmp_2 = (2. - nu( j )) / G( j );
          double G_eff = 1. / (tmp_1 + tmp_2);
          // # Eq 12 [1]
          double kt = 8. * G_eff * sqrt(R_eff * overlap);
          double S_t = kt;
          double eta_t = -2 * sqrt(5/6) * beta * sqrt(S_t * m_eff);

          // normal force
          double fn =  kn * pow(overlap, 1.5);
          double fn_x = fn * nij_x - eta_n * vn_x;
          double fn_y = fn * nij_y - eta_n * vn_y;
          double fn_z = fn * nij_z - eta_n * vn_z;

          // Tangential force
          auto tangential_frc_ss_x_i = 0.;
          auto tangential_frc_ss_y_i = 0.;
          auto tangential_frc_ss_z_i = 0.;
          // Incremenet the tangential force and add it to the total force
          // check if there is relative motion
          double vij_magn = pow(vel_ij[0]*vel_ij[0] + vel_ij[1]*vel_ij[1] + vel_ij[2]*vel_ij[2], 0.5);
          if (vij_magn < 1e-12) {
            // make the tangential displacement
            tangential_disp_ss_x ( i, found_at ) = 0.;
            tangential_disp_ss_y ( i, found_at ) = 0.;
            tangential_disp_ss_z ( i, found_at ) = 0.;

            // // make the tangential tangential force
            // aosoa_tng_frc ( i, found_at, 0 ) = 0.;
            // aosoa_tng_frc ( i, found_at, 1 ) = 0.;
            // aosoa_tng_frc ( i, found_at, 2 ) = 0.;

            // // # repulsive force
            // d_fn_x[t2] = fn_x;
            // d_fn_y[t2] = fn_y;
            // d_fn_z[t2] = fn_z;
          }
          else{
            // the tangential vector is
            double tx_tmp = vel_ij[0] - nij_x * vij_dot_nij;
            double ty_tmp = vel_ij[1] - nij_y * vij_dot_nij;
            double tz_tmp = vel_ij[2] - nij_z * vij_dot_nij;

            double ti_magn = pow(tx_tmp*tx_tmp + ty_tmp*ty_tmp + tz_tmp*tz_tmp, 0.5);

            double ti_x = 0.;
            double ti_y = 0.;
            double ti_z = 0.;

            if (ti_magn > 1e-12){
              ti_x = tx_tmp / ti_magn;
              ti_y = ty_tmp / ti_magn;
              ti_z = tz_tmp / ti_magn;
            }

            // this is correct
            double delta_lt_x_star = tangential_disp_ss_x ( i, found_at ) + vel_ij[0] * dt;
            double delta_lt_y_star = tangential_disp_ss_y ( i, found_at ) + vel_ij[1] * dt;
            double delta_lt_z_star = tangential_disp_ss_z ( i, found_at ) + vel_ij[2] * dt;

            double delta_lt_dot_ti = (delta_lt_x_star * ti_x +
                                      delta_lt_y_star * ti_y +
                                      delta_lt_z_star * ti_z);

            tangential_disp_ss_x ( i, found_at ) = delta_lt_dot_ti * ti_x;
            tangential_disp_ss_y ( i, found_at ) = delta_lt_dot_ti * ti_y;
            tangential_disp_ss_z ( i, found_at ) = delta_lt_dot_ti * ti_z;

            double ft_x_star = -kt * tangential_disp_ss_x ( i, found_at ) - eta_t * vt_x;
            double ft_y_star = -kt * tangential_disp_ss_y ( i, found_at ) - eta_t * vt_y;
            double ft_z_star = -kt * tangential_disp_ss_z ( i, found_at ) - eta_t * vt_z;

            double ft_magn = pow(ft_x_star*ft_x_star + ft_y_star*ft_y_star + ft_z_star*ft_z_star, 0.5);
            double fn_magn = pow(fn_x*fn_x + fn_y*fn_y + fn_z*fn_z, 0.5);

            double ft_magn_star = ft_magn;
            if (ft_magn_star > friction_pp * fn_magn){
              ft_magn_star = friction_pp * fn_magn;
            }

            // compute the tangential force, by equation 27
            tangential_frc_ss_x_i = -ft_magn_star * ti_x;
            tangential_frc_ss_y_i = -ft_magn_star * ti_y;
            tangential_frc_ss_z_i = -ft_magn_star * ti_z;

            double modified_delta_lt_x = -tangential_frc_ss_x_i / kt;
            double modified_delta_lt_y = -tangential_frc_ss_y_i / kt;
            double modified_delta_lt_z = -tangential_frc_ss_z_i / kt;

            double lt_magn = pow(modified_delta_lt_x*modified_delta_lt_x + modified_delta_lt_y*modified_delta_lt_y +
                                 modified_delta_lt_z*modified_delta_lt_z, 0.5);

            if (lt_magn > 1e-12){
              tangential_disp_ss_x ( i, found_at ) = modified_delta_lt_x / lt_magn;
              tangential_disp_ss_y ( i, found_at ) = modified_delta_lt_y / lt_magn;
              tangential_disp_ss_z ( i, found_at ) = modified_delta_lt_z / lt_magn;
            }
            else {
              tangential_disp_ss_x ( i, found_at ) = 0.;
              tangential_disp_ss_y ( i, found_at ) = 0.;
              tangential_disp_ss_z ( i, found_at ) = 0.;
            }
          }


          // Add force to the particle i due to contact with particle j
          force( i, 0 ) += fn_x + tangential_frc_ss_x_i ;
          force( i, 1 ) += fn_y + tangential_frc_ss_y_i ;
          force( i, 2 ) += fn_z + tangential_frc_ss_z_i ;
        }

      };

    Kokkos::RangePolicy<ExecutionSpace> policy(0, u.size());


    Cabana::neighbor_parallel_for( policy,
                                   force_full,
                                neigh_list,
                                     Cabana::FirstNeighborsTag(),
                                     neigh_op_tag,
                                     "CabanaDEM::ForceFull" );
    Kokkos::fence();
    }

    template <class ParticleType, class NeighListType>
    void update_tangential_contacts(ParticleType& particles,
                                 const NeighListType& neigh_list){
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
      auto tangential_disp_ss_x = particles.sliceTangentialDispSSx();
      auto tangential_disp_ss_y = particles.sliceTangentialDispSSy();
      auto tangential_disp_ss_z = particles.sliceTangentialDispSSz();
      auto tangential_disp_idx = particles.sliceTangentialDispIdx();
      auto total_no_tangential_contacts = particles.sliceTotalNoTangentialContacts();

      auto update_tangential_contacts_lambda_func = KOKKOS_LAMBDA( const int i )
        {
          int count = 0;
          int k = 0;
          int idx_total_ctcs = total_no_tangential_contacts ( i );
          int last_idx_tmp = total_no_tangential_contacts ( i ) - 1;
          int sidx = -1;
          // loop over all the contacts of particle d_idx
          while (count < idx_total_ctcs){
            // The index of the particle with which
            // d_idx in contact is
            sidx = tangential_disp_idx ( i, k );
            if (sidx == -1){
              break;
            }
            else {
              double pos_i[3] = {x( i, 0 ),
                                 x( i, 1 ),
                                 x( i, 2 )};

              double pos_j[3] = {x( sidx, 0 ),
                                 x( sidx, 1 ),
                                 x( sidx, 2 )};

              double pos_ij[3] = {x( i, 0 ) - x( sidx, 0 ),
                                  x( i, 1 ) - x( sidx, 1 ),
                                  x( i, 2 ) - x( sidx, 2 )};

              // squared distance
              double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
              // distance between i and j
              double rij = sqrt(r2ij);
              // Find the overlap amount
              double overlap =  rad ( i ) + rad ( sidx ) - rij;

              if (overlap <= 0.) {
                // if the swap index is the current index then
                // simply make it to null contact.
                if (k == last_idx_tmp){
                  tangential_disp_idx( i, k ) = -1;
                  tangential_disp_ss_x( i, k ) = 0.;
                  tangential_disp_ss_y( i, k ) = 0.;
                  tangential_disp_ss_z( i, k ) = 0.;
                }
                else {
                  // swap the current tracking index with the final
                  // contact index
                  tangential_disp_idx( i, k ) = tangential_disp_idx( i, last_idx_tmp );
                  tangential_disp_idx( i, last_idx_tmp ) = -1;

                  // swap tangential x displacement
                  tangential_disp_ss_x( i, k ) = tangential_disp_ss_x( i, last_idx_tmp );
                  tangential_disp_ss_x( i, last_idx_tmp ) = 0.;

                  // swap tangential y displacement
                  tangential_disp_ss_y( i, k ) = tangential_disp_ss_y( i, last_idx_tmp );
                  tangential_disp_ss_y( i, last_idx_tmp ) = 0.;

                  // swap tangential z displacement
                  tangential_disp_ss_z( i, k ) = tangential_disp_ss_z( i, last_idx_tmp );
                  tangential_disp_ss_z( i, last_idx_tmp ) = 0.;

                  // decrease the last_idx_tmp, since we swapped it to
                  // -1
                  last_idx_tmp -= 1;
                }

                // decrement the total contacts of the particle
                total_no_tangential_contacts ( i ) -= 1;
              }
              else
                {
                  k = k + 1;
                }
            }
            count += 1;
          }
        };
      Kokkos::RangePolicy<ExecutionSpace> policy( 0, x.size() );
      Kokkos::parallel_for( "CabanaDEM:Force:UpdateTngCnts", policy,
                         update_tangential_contacts_lambda_func );
    }

    template <class ParticleType, class InfiniteWallType>
    void computeForceFullParticleInfiniteWall(ParticleType& particles,
                                              InfiniteWallType& wall, double dt)
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
      auto tangential_disp_sw_x = particles.sliceTangentialDispSWx();
      auto tangential_disp_sw_y = particles.sliceTangentialDispSWy();
      auto tangential_disp_sw_z = particles.sliceTangentialDispSWz();

      auto x_w = wall.slicePosition();
      auto u_w = wall.sliceVelocity();
      auto omega_w = wall.sliceOmega();
      auto normal_w = wall.sliceNormal();
      auto E_w = wall.sliceYoungsMod();
      auto nu_w = wall.slicePoissonsRatio();
      auto G_w = wall.sliceShearMod();

      auto force_full = KOKKOS_LAMBDA( const int i )
        {
          for (int j=0; j < x_w.size(); j++){
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

            double pos_j[3] = {x_w( j, 0 ),
              x_w( j, 1 ),
              x_w( j, 2 )};

            double pos_ij[3] = {x( i, 0 ) - x_w( j, 0 ),
              x( i, 1 ) - x_w( j, 1 ),
              x( i, 2 ) - x_w( j, 2 )};

            // squared distance
            double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
            // distance between i and j
            double rij = sqrt(r2ij);

            const double mass_i = m( i );

            // normal vector passing from j to i
            double nij_x = normal_w( j, 0 );
            double nij_y = normal_w( j, 1 );
            double nij_z = normal_w( j, 2 );

            // Find the overlap amount
            double tmp = pos_ij[0] * nij_x  + pos_ij[1] * nij_y + pos_ij[2] * nij_z;
            double overlap =  rad ( i ) - tmp;

            double a_i = rad ( i ) - overlap / 2.;
            double vel_i[3] = {0., 0., 0.};

            vel_i[0] = u ( i, 0 ) +
            (omega( i, 1 ) * nij_z - omega( i, 2 ) * nij_y) * a_i;

            vel_i[1] = u ( i, 1 ) +
            (omega( i, 2 ) * nij_x - omega( i, 0 ) * nij_z) * a_i;

            vel_i[2] = u ( i, 2 ) +
            (omega( i, 0 ) * nij_y - omega( i, 1 ) * nij_x) * a_i;

            double vel_j[3] = {0., 0., 0.};

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

            /*
              ====================================
              End: common to all equations in SPH.
              ====================================
            */

            // std::cout << "inside overlap" << std::endl;
            // find the force if the particles are overlapping
            if (overlap > 0.) {
              // std::cout << "inside overlap" << std::endl;

              // Compute stiffness
              // effective Young's modulus
              double tmp_1 = (1. - (nu( i )*nu( i ))) / E( i );
              double tmp_2 = (1. - (nu_w( j )*nu_w( j ))) / E_w( j );
              double E_eff = 1. / (tmp_1 + tmp_2);
              double tmp_3 = 1. / rad( i );
              double tmp_4 = 0.;
              double R_eff = 1. / (tmp_3 + tmp_4);
              // # Eq 4 [1]
              double kn = 4. / 3. * E_eff * sqrt(R_eff);

              // # compute damping coefficient
              tmp_1 = log(cor_pw);
              // std::cout << "log value is" << tmp_1 << std::endl;
              // tmp_1 = 0.;
              tmp_2 = tmp_1 * tmp_1 + M_PI * M_PI;
              double beta = tmp_1 / sqrt(tmp_2);
              double S_n = 2. * E_eff * sqrt(R_eff * overlap);
              tmp_1 = 1. / m( i );
              tmp_2 = 0.;
              double m_eff = 1. / (tmp_1 + tmp_2);
              double eta_n = -2. * sqrt(5./6.) * beta * sqrt(S_n * m_eff);

              // normal force
              double fn =  kn * pow(overlap, 1.5);
              double fn_x = fn * nij_x - eta_n * vn_x;
              double fn_y = fn * nij_y - eta_n * vn_y;
              double fn_z = fn * nij_z - eta_n * vn_z;

              // #################################
              // # tangential force computation  #
              // #################################
              // # if the particle is not been tracked then assign an index in
              // # tracking history.
              double ft_x = 0.;
              double ft_y = 0.;
              double ft_z = 0.;
              // # tangential velocity
              double vij_magn = sqrt(vel_ij[0]*vel_ij[0] + vel_ij[1]*vel_ij[1] + vel_ij[2]*vel_ij[2]);

              if (vij_magn < 1e-12){
                tangential_disp_sw_x( i, j )	= 0.;
                tangential_disp_sw_y( i, j )	= 0.;
                tangential_disp_sw_z( i, j )	= 0.;
              }
              else{
                // # print("inside")
                // # project tangential spring on the current plane normal
                tangential_disp_sw_x( i, j ) += vt_x * dt;
                tangential_disp_sw_y( i, j ) += vt_y * dt;
                tangential_disp_sw_z( i, j ) += vt_z * dt;

                // # Compute the tangential stiffness
                tmp_1 = (2. - nu( i )) / G( i );
                tmp_2 = (2. - nu_w( j )) / G_w( j );
                double G_eff = 1. / (tmp_1 + tmp_2);
                // # Eq 12 [1]
                double kt = 8. * G_eff * sqrt(R_eff * overlap);
                double S_t = kt;
                double eta_t = -2 * sqrt(5/6) * beta * sqrt(S_t * m_eff);

                double ft_x_star = -kt * tangential_disp_sw_x( i, j ) - eta_t * vt_x;
                double ft_y_star = -kt * tangential_disp_sw_y( i, j ) - eta_t * vt_y;
                double ft_z_star = -kt * tangential_disp_sw_z( i, j ) - eta_t * vt_z;

                double ft_magn = sqrt(ft_x_star*ft_x_star + ft_y_star*ft_y_star + ft_z_star*ft_y_star);

                double ti_x = 0.;
                double ti_y = 0.;
                double ti_z = 0.;

                if (ft_magn > 1e-12){
                  ti_x = ft_x_star / ft_magn;
                  ti_y = ft_y_star / ft_magn;
                  ti_z = ft_z_star / ft_magn;
                }

                double fn_magn = sqrt(fn_x*fn_x + fn_y*fn_y + fn_z*fn_z);

                double ft_magn_star = std::min(friction_pw * fn_magn, ft_magn);

                // # compute the tangential force, by equation 17 (Lethe)
                ft_x = ft_magn_star * ti_x;
                ft_y = ft_magn_star * ti_y;
                ft_z = ft_magn_star * ti_z;

                // # Add damping to the limited force
                ft_x += eta_t * vt_x;
                ft_y += eta_t * vt_y;
                ft_z += eta_t * vt_z;

                // # reset the spring length
                tangential_disp_sw_x( i, j )	= -ft_x / kt;
                tangential_disp_sw_y( i, j )	= -ft_y / kt;
                tangential_disp_sw_z( i, j )	= -ft_z / kt;

              }
              // Add force to the particle i due to contact with particle j
              force( i, 0 ) += fn_x + ft_x;
              force( i, 1 ) += fn_y + ft_y;
              force( i, 2 ) += fn_z + ft_z;

              // # torque = n cross F
              torque( i, 0 ) += (nij_y * ft_z - nij_z * ft_y) * a_i;
              torque( i, 1 ) += (nij_z * ft_x - nij_x * ft_z) * a_i;
              torque( i, 2 ) += (nij_x * ft_y - nij_y * ft_x) * a_i;
            }
            else {
              tangential_disp_sw_x( i, j ) = 0.;
              tangential_disp_sw_y( i, j ) = 0.;
              tangential_disp_sw_z( i, j ) = 0.;
            }
          }

        };

      Kokkos::RangePolicy<ExecutionSpace> policy( 0, u.size() );
      Kokkos::parallel_for( "CabanaDEM::Force::ParticleInfiniteWall", policy,
                            force_full );
    }
  };



  /******************************************************************************
  Force free functions.
  ******************************************************************************/
  template <class ForceType, class ParticleType, class NeighListType,
            class ParallelType>
  void computeForceParticleParticle( ForceType& force, ParticleType& particles,
                                     const NeighListType& neigh_list,
                                     const ParallelType& neigh_op_tag,
                                     double dt )
  {
    force.makeForceTorqueZeroOnParticle( particles );
    force.update_tangential_contacts( particles, neigh_list );
    force.computeForceFullParticleParticle( particles, neigh_list, neigh_op_tag, dt );
  }

  template <class ForceType, class ParticleType, class InfiniteWallType, class NeighListType,
            class ParallelType>
  void computeForceParticleParticleInfiniteWall( ForceType& force, ParticleType& particles,
                                                 InfiniteWallType& wall,
                                                 const NeighListType& neigh_list,
                                                 const ParallelType& neigh_op_tag,
                                                 double dt )
  {
    force.makeForceTorqueZeroOnParticle( particles );
    force.update_tangential_contacts( particles, neigh_list );
    force.computeForceFullParticleParticle( particles, neigh_list, neigh_op_tag, dt );
    force.computeForceFullParticleInfiniteWall( particles, wall, dt );
  }
}

#endif
