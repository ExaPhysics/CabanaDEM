from math import (sqrt, asin, sin, cos)
import numpy as np
from numpy import sqrt, fabs
from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)
from textwrap import dedent
from compyle.api import declare
from pysph.sph.integrator_step import IntegratorStep
from pysph.base.utils import get_particle_array


def add_properties_stride(pa, stride=1, *props):
    for prop in props:
        pa.add_property(name=prop, stride=stride)


def get_particle_array_bonded_dem(constants=None, **props):
    bonded_dem_props = ['x0', 'y0', 'u0', 'v0', 'rad', 'fx', 'fy', 'fz']
    consts = {
        'no_bonds_limits': np.array([8], dtype='int'),
        'criterion_dist': -1.
    }

    if constants:
        consts.update(constants)
    pa = get_particle_array(constants=consts,
                            additional_props=bonded_dem_props, **props)

    # contact indices
    pa.add_property('cnt_idxs', stride=consts['no_bonds_limits'], type='int')
    # distance between the particles at the initiation  of the contacts
    pa.add_property('delta_equi', stride=consts['no_bonds_limits'])
    # each particle contact limits
    pa.add_property('cnt_limits', stride=2, type='int')
    # each particle total number of contacts
    pa.add_property('tot_cnts', type='int')

    # set the contacts to default values
    pa.cnt_idxs[:] = -1
    pa.cnt_limits[:] = 0
    pa.tot_cnts[:] = 0

    set_contacts(pa, pa.criterion_dist[0])

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])

    return pa


class SetContactsBondedDEM(Equation):
    def __init__(self, dest, sources, criterion_dist=0.3):
        super(SetContactsBondedDEM, self).__init__(dest, sources)
        self.criterion_dist = criterion_dist

    def loop(self, d_idx, d_x, d_y, d_rad, d_cnt_idxs, d_cnt_limits,
             d_tot_cnts, d_no_bonds_limits, d_delta_equi, s_idx, s_x, s_y,
             s_rad, RIJ):
        i = declare('int')
        if d_idx != s_idx:
            if RIJ < self.criterion_dist:
                # add the contact index at the end of the list
                i = d_idx * d_no_bonds_limits[0] + d_tot_cnts[d_idx]
                d_cnt_idxs[i] = s_idx
                d_delta_equi[i] = RIJ

                # increment the total number of contacts
                d_tot_cnts[d_idx] += 1

                # set the no of bonds limit
                d_cnt_limits[2 * d_idx] = d_idx * d_no_bonds_limits[0]
                d_cnt_limits[2 * d_idx + 1] = (
                    d_idx * d_no_bonds_limits[0] + d_tot_cnts[d_idx])


def set_contacts(pa, criterion_dist):
    assert criterion_dist > 0., "the criterion_dist has to be positive"
    assert pa.no_bonds_limits > 0, "the criterion_dist has to be positive"

    equations = [
        Group(
            equations=[SetContactsBondedDEM(dest=pa.name, sources=[pa.name])])
    ]

    sph_eval = SPHEvaluator(arrays=[pa], equations=equations, dim=2,
                            kernel=CubicSpline(dim=2))

    sph_eval.evaluate(0.1, 0.1)


class GTVFStepSpringBondModel(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_wx, d_wy, d_wz,
               d_fx, d_fy, d_fz, d_tor_x, d_tor_y, d_tor_z, d_m, d_moi, dt):
        dtb2 = dt / 2.

        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_fx[d_idx] / d_m[d_idx]
        d_v[d_idx] += dtb2 * d_fy[d_idx] / d_m[d_idx]
        d_w[d_idx] += dtb2 * d_fz[d_idx] / d_m[d_idx]

        d_wx[d_idx] += dtb2 * d_tor_x[d_idx] / d_moi[d_idx]
        d_wy[d_idx] += dtb2 * d_tor_y[d_idx] / d_moi[d_idx]
        d_wz[d_idx] += dtb2 * d_tor_z[d_idx] / d_moi[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_wx, d_wy, d_wz,
               d_theta_x, d_theta_y, d_theta_z, dt):
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]

        d_theta_x[d_idx] += dt * d_wx[d_idx]
        d_theta_y[d_idx] += dt * d_wy[d_idx]
        d_theta_z[d_idx] += dt * d_wz[d_idx]

    def stage3(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_wx, d_wy, d_wz,
               d_fx, d_fy, d_fz, d_tor_x, d_tor_y, d_tor_z, d_m, d_moi, dt):
        dtb2 = dt / 2.

        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_fx[d_idx] / d_m[d_idx]
        d_v[d_idx] += dtb2 * d_fy[d_idx] / d_m[d_idx]
        d_w[d_idx] += dtb2 * d_fz[d_idx] / d_m[d_idx]

        d_wx[d_idx] += dtb2 * d_tor_x[d_idx] / d_moi[d_idx]
        d_wy[d_idx] += dtb2 * d_tor_y[d_idx] / d_moi[d_idx]
        d_wz[d_idx] += dtb2 * d_tor_z[d_idx] / d_moi[d_idx]


class ResetForce(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz,
                   d_tor_x, d_tor_y, d_tor_z):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.
        d_tor_x[d_idx] = 0.
        d_tor_y[d_idx] = 0.
        d_tor_z[d_idx] = 0.


class ParticleDampingForce(Equation):
    def __init__(self, dest, sources, gamma_b=0.1):
        self.gamma_b = gamma_b
        super(ParticleDampingForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz,
                   d_tor_x, d_tor_y, d_tor_z, d_u, d_v, d_w,
                   d_wx, d_wy, d_wz):
        v_magn = (d_u[d_idx]**2. + d_v[d_idx]**2. + d_w[d_idx]**2.)**0.5
        omega_magn = (d_wx[d_idx]**2. + d_wy[d_idx]**2. + d_wz[d_idx]**2.)**0.5
        if v_magn > 1e-12:
            fac = self.gamma_b / v_magn * 1e7
            d_fx[d_idx] -= d_u[d_idx] * fac
            d_fy[d_idx] -= d_v[d_idx] * fac
            d_fz[d_idx] -= d_w[d_idx] * fac

        if omega_magn > 1e-12:
            fac = self.gamma_b / omega_magn
            d_tor_x[d_idx] -= d_tor_x[d_idx] * d_wx[d_idx] * fac
            d_tor_y[d_idx] -= d_tor_y[d_idx] * d_wy[d_idx] * fac
            d_tor_z[d_idx] -= d_tor_z[d_idx] * d_wz[d_idx] * fac


class FixStaticParticles(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz,
                   d_tor_x, d_tor_y, d_tor_z, d_u, d_v, d_w,
                   d_wx, d_wy, d_wz, d_is_static):
        if d_is_static[d_idx] == 1.:
            d_fx[d_idx] = 0.
            d_fy[d_idx] = 0.
            d_fz[d_idx] = 0.

            d_tor_x[d_idx] = 0.
            d_tor_y[d_idx] = 0.
            d_tor_z[d_idx] = 0.

            d_u[d_idx] = 0.
            d_v[d_idx] = 0.
            d_w[d_idx] = 0.

            d_wx[d_idx] = 0.
            d_wy[d_idx] = 0.
            d_wz[d_idx] = 0.


class BondedDEMInterParticleLinearForce(Equation):
    def initialize(self, d_idx, d_x, d_y, d_z, d_rad, d_cnt_idxs, d_cnt_limits,
                   d_tot_cnts, d_no_bonds_limits, d_u, d_v, d_w,
                   d_wy, d_wx, d_wz,
                   d_init_length_bond,
                   d_ft_x_bond, d_ft_y_bond, d_ft_z_bond,
                   d_tor_x_bond, d_tor_y_bond, d_tor_z_bond, d_fx, d_fy, d_fz,
                   d_tor_x, d_tor_y, d_tor_z, d_m, d_moi, dt):
        i, p, q, sidx = declare('int', 4)
        # particle d_idx has its neighbours information in d_cnt_idxs
        # The range of such is
        p = d_cnt_limits[2 * d_idx]
        q = d_cnt_limits[2 * d_idx + 1]

        # now loop over the neighbours and find the force on particle d_idx
        k_n_bond = 1e7
        k_t_bond = 1e5
        k_tor = 1e7
        k_ben = 1e5

        # now loop over the neighbours and find the force on particle d_idx
        for i in range(p, q):
            # ======================
            # find the contact point
            # ======================
            # unit vector passing from alpha to beta (i to j)
            sidx = d_cnt_idxs[i]
            pos_ij_x = d_x[sidx] - d_x[d_idx]
            pos_ij_y = d_y[sidx] - d_y[d_idx]
            pos_ij_z = d_z[sidx] - d_z[d_idx]
            rij = ((pos_ij_x)**2. + (pos_ij_y)**2. + (pos_ij_z)**2.)**(0.5)
            n_ij_x = pos_ij_x / rij
            n_ij_y = pos_ij_y / rij
            n_ij_z = pos_ij_z / rij

            # find the distance till middle point of the contact
            gap = rij - (d_rad[d_idx] + d_rad[sidx])
            dist_c = d_rad[d_idx] + gap / 2.
            x_c = d_x[d_idx] + n_ij_x * dist_c
            y_c = d_y[d_idx] + n_ij_y * dist_c
            z_c = d_z[d_idx] + n_ij_z * dist_c

            # find the contact point vector of the d_idx (a is alpha)
            x_ac = x_c - d_x[d_idx]
            y_ac = y_c - d_y[d_idx]
            z_ac = z_c - d_z[d_idx]

            r_ac = (x_ac**2. + y_ac**2. + z_ac**2.)**0.5

            # find the contact point vector of the s_idx (b is beta)
            x_bc = x_c - d_x[sidx]
            y_bc = y_c - d_y[sidx]
            z_bc = z_c - d_z[sidx]

            # find the velocity of particle d_idx (alpha, i) at the contact
            # point
            u_ac = d_u[d_idx] + (d_wy[d_idx] * z_ac + d_wz[d_idx] * y_ac)
            v_ac = d_v[d_idx] + (d_wx[d_idx] * z_ac + d_wz[d_idx] * x_ac)
            w_ac = d_w[d_idx] + (d_wx[d_idx] * y_ac + d_wy[d_idx] * x_ac)

            u_bc = d_u[sidx] + (d_wy[sidx] * z_bc + d_wz[sidx] * y_bc)
            v_bc = d_v[sidx] + (d_wx[sidx] * z_bc + d_wz[sidx] * x_bc)
            w_bc = d_w[sidx] + (d_wx[sidx] * y_bc + d_wy[sidx] * x_bc)

            # find the relative linear velocity of particle alpha about
            # beta
            u_cr = u_ac - u_bc
            v_cr = v_ac - v_bc
            w_cr = w_ac - w_bc

            # Calculate normal and tangential components of relative velocity
            vn = u_cr * n_ij_x + v_cr * n_ij_y + w_cr * n_ij_z
            vn_x = vn * n_ij_x
            vn_y = vn * n_ij_y
            vn_z = vn * n_ij_z

            vt_x = u_cr - vn_x
            vt_y = v_cr - vn_y
            vt_z = w_cr - vn_z

            # find the relative angular velocity of particle alpha about
            # beta
            wx_cr = d_wx[d_idx] - d_wx[sidx]
            wy_cr = d_wy[d_idx] - d_wy[sidx]
            wz_cr = d_wz[d_idx] - d_wz[sidx]
            wn = wx_cr * n_ij_x + wy_cr * n_ij_y + wz_cr * n_ij_z
            wn_x = wn * n_ij_x
            wn_y = wn * n_ij_y
            wn_z = wn * n_ij_z

            wt_x = wx_cr - wn_x
            wt_y = wy_cr - wn_y
            wt_z = wz_cr - wn_z

            # =========================
            # Compute the normal force
            # =========================
            overlap_n = rij - d_init_length_bond[i]
            fn_x = -k_n_bond * overlap_n * n_ij_x
            fn_y = -k_n_bond * overlap_n * n_ij_y
            fn_z = -k_n_bond * overlap_n * n_ij_z

            # minvel = 1e-5 * min(d_rad[d_idx], d_rad[s_idx]) /dt
            # dvX = 0.01 * fn_x*dt
            # dvY = 0.01 * fn_x*dt
            # dvZ = 0.01 * fn_x*dt
            # multiplierX = (vrel_n[0] >= 0.0) ? 1.0 : -1.0
            # multiplierY = (vrel_n[1] >= 0.0) ? 1.0 : -1.0
            # multiplierZ = (vrel_n[2] >= 0.0) ? 1.0 : -1.0

            # fn_damped_x = fn_x - beta_bond*fabs(fn_x) * multiplierX
            # fn_damped_x = fn_x - beta_bond*fabs(fn_x) * multiplierY
            # fn_damped_x = fn_x - beta_bond*fabs(fn_x) * multiplierZ
            # ==============================
            # Compute the normal force ends
            # ==============================

            # ==============================
            # Compute the tangential force
            # ==============================
            ft_x = d_ft_x_bond[i]
            ft_y = d_ft_y_bond[i]
            ft_z = d_ft_z_bond[i]

            ft_dp_n = ft_x * n_ij_x + ft_y * n_ij_y + ft_z * n_ij_z
            ft_norm_x = ft_dp_n * n_ij_x
            ft_norm_y = ft_dp_n * n_ij_y
            ft_norm_z = ft_dp_n * n_ij_z

            # rotate it to the current plane
            ft_x -= ft_norm_x
            ft_y -= ft_norm_y
            ft_z -= ft_norm_z

            # add the increment
            tmp = k_t_bond * dt
            ft_x += -vt_x * tmp
            ft_y += -vt_y * tmp
            ft_z += -vt_z * tmp

            # dvX = 0.01*ft[0]*dt
            # dvY = 0.01*ft[1]*dt
            # dvZ = 0.01*ft[2]*dt
            # multiplierX = (vrel_t[0] >= 0.0) ? 1.0 : -1.0
            # multiplierY = (vrel_t[1] >= 0.0) ? 1.0 : -1.0
            # multiplierZ = (vrel_t[2] >= 0.0) ? 1.0 : -1.0
            # ft_damped[0] = ft[0] - beta_bond*fabs(ft[0])*multiplierX
            # ft_damped[1] = ft[1] - beta_bond*fabs(ft[1])*multiplierY
            # ft_damped[2] = ft[2] - beta_bond*fabs(ft[2])*multiplierZ

            # bond_ft_x (i, j) = ft_damped[0];
            # bond_ft_y (i, j) = ft_damped[1];
            # bond_ft_z (i, j) = ft_damped[2];

            # ==================================
            # Compute the tangential force ends
            # ==================================

            # # add the force and moment to the global force of particle i
            # d_fx[d_idx] += d_ft_x_bond[i]
            # d_fy[d_idx] += d_ft_y_bond[i]
            # d_fz[d_idx] += d_ft_z_bond[i]
            # d_tor_x[d_idx] += d_tor_x_bond[i]
            # d_tor_y[d_idx] += d_tor_y_bond[i]
            # d_tor_z[d_idx] += d_tor_z_bond[i]


class SpringBondModel(Scheme):
    def __init__(self, solids, dim,
                 kr=1e8, kf=1e5, fric_coeff=0.0, gx=0., gy=0., gz=0.):
        self.solids = solids

        self.dim = dim

        self.debug = False

        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff

        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--kr-stiffness", action="store",
                           dest="kr", default=1e8,
                           type=float,
                           help="Repulsive spring stiffness")

        group.add_argument("--kf-stiffness", action="store",
                           dest="kf", default=1e3,
                           type=float,
                           help="Tangential spring stiffness")

        group.add_argument("--fric-coeff", action="store",
                           dest="fric_coeff", default=0.0,
                           type=float,
                           help="Friction coefficient")

    def consume_user_options(self, options):
        _vars = ['kr', 'kf', 'fric_coeff']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.sph.wc.gtvf import GTVFIntegrator
        from pysph.base.kernels import QuinticSpline
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        bodystep = GTVFStepSpringBondModel()
        integrator_cls = GTVFIntegrator

        for body in self.solids:
            if body not in steppers:
                steppers[body] = bodystep

        cls = integrator_cls
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim,
                             integrator=integrator,
                             kernel=kernel,
                             **kw)

    def get_equations(self):
        # ==============================
        # Stage 1 equations
        # ==============================
        stage1 = []
        # ==============================
        # Stage 2 equations
        # ==============================
        stage2 = []
        #######################
        # Handle rigid bodies #
        #######################
        g5 = []
        for name in self.solids:
            g5.append(
                ResetForce(dest=name, sources=None))

            g5.append(
                BondedDEMInterParticleLinearForce(
                    dest=name,
                    sources=None))

            g5.append(
                ParticleDampingForce(
                    dest=name,
                    sources=None))

            g5.append(
                FixStaticParticles(
                    dest=name,
                    sources=None))

        stage2.append(Group(equations=g5, real=False))

        return MultiStageEquations([stage1, stage2])

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties

        pas = dict([(p.name, p) for p in particles])

        for solid in self.solids:
            pa = pas[solid]

            # properties to find the find on the rigid body by
            # Mofidi, Drescher, Emden, Teschner
            add_properties_stride(pa, pa.no_bonds_limits[0],
                                  'init_length_bond',
                                  'ft_x_bond', 'ft_y_bond', 'ft_z_bond',
                                  'fx_bond', 'fy_bond', 'fz_bond',
                                  'tor_x_bond', 'tor_y_bond',
                                  'tor_z_bond')

            add_properties(pa, 'fx', 'fy', 'fz', 'tor_x', 'tor_y', 'tor_z',
                           'theta_x', 'theta_y', 'theta_z', 'moi',
                           'wx', 'wy', 'wz', 'is_static')
            pa.is_static[:] = 0.

            # contact indices
            pa.add_property('cnt_idxs', stride=pa.no_bonds_limits[0], type='int')
            # distance between the particles at the initiation  of the contacts
            pa.add_property('delta_equi', stride=pa.no_bonds_limits[0])
            # each particle contact limits
            pa.add_property('cnt_limits', stride=2, type='int')
            # each particle total number of contacts
            pa.add_property('tot_cnts', type='int')

            # set the contacts to default values
            pa.cnt_idxs[:] = -1
            pa.cnt_limits[:] = 0
            pa.tot_cnts[:] = 0

            set_contacts(pa, pa.criterion_dist[0])

    def get_solver(self):
        return self.solver
