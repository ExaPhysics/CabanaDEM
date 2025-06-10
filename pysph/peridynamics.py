# # ========================================================
# Follow paper [1].  Paper [2] also has good explanation about
# bond-based peridynamics which we have implemented in this work.  For
# code look into [1c] and [2c]. Here [2c] follows paper [2] equations
# for bond-based model.



# [1] A coupled peridynamics--smoothed particle hydrodynamics model for fracture analysis of fluid--structure interactions
# [2] Guan, Jinwei, and Li Guo. "A unified bond–based peridynamic model without limitation of Poisson's ratio." Applied Mathematical Modelling 128 (2024): 609-629. https://www.sciencedirect.com/science/article/pii/S0307904X24000143

# [1c] https://github.com/ORNL/PDMATLAB2D.
# [2c] https://github.com/PeriHub/PeriLab.jl

# *c implies code
# # ========================================================

# ## Bond based peridynamics

# ### From paper [1], the force formulation is given as:

# $$
# \textbf{F} = \mu c (\boldsymbol{\xi}) s (\boldsymbol{\eta}, \boldsymbol{\xi}) \frac{\boldsymbol{\eta} + \boldsymbol{\xi}}{||\boldsymbol{\eta} + \boldsymbol{\xi}||}
# $$

# Essentially here, we are multiplying the foce magnitude, $\mu c
# (\boldsymbol{\xi}) s (\boldsymbol{\eta}, \boldsymbol{\xi})$ with the force direction
# which is the current vector passing from particle $i$ to particle $j$.

# Few points to note here are, \boldsymbol{\xi} is defined as the vector passing from particle $i$
# to particle $j$ in the reference frame (i.e., in the initial configuration), while
# $\boldsymbol{\eta} + \boldsymbol{\xi}$ is in the current configuration.

# $s (\boldsymbol{\eta}, \boldsymbol{\xi})$ is the bond length ratio, defined as

# $$
# s = \frac{||\boldsymbol{\eta} + \boldsymbol{\xi}|| - ||\boldsymbol{\xi}||}{||\boldsymbol{\xi}||}
# $$
from math import (sqrt, asin, sin, cos)
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from numpy import sqrt, fabs
from mpl_toolkits.mplot3d import Axes3D

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.base.kernels import QuinticSpline
from textwrap import dedent
from compyle.api import declare
from pysph.sph.integrator_step import IntegratorStep
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import Integrator


def scatter_bonded_particles(pa, must_plot_indices, folder_name, dim=2, show_plot=False):
    # Delete the folder if it exists and then recreate it
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    # Select some indices from particles
    random_indices = np.random.choice(np.arange(len(pa.x)), size=10, replace=False)

    # Merge with the must_plot_indices, ensuring no duplicates
    indices = np.union1d(random_indices, must_plot_indices)

    # Loop over the selected indices and plot for each particle
    for i, index in enumerate(indices):
        # Access the particle's x, y, and z coordinates
        x_val = pa.x[index]
        y_val = pa.y[index]
        z_val = pa.z[index]  # Assuming `z` coordinate exists

        # Create a new figure
        fig = plt.figure(figsize=(8, 8))

        if dim == 3:
            # Plotting in 3D
            ax = fig.add_subplot(111, projection='3d')

            # Plot all particles in blue
            ax.scatter(pa.x, pa.y, pa.z, c='blue', alpha=0.5, label="All particles")

            # Get the bonded particles for the current particle
            bond_indices_i = pa.cnt_idxs[pa.cnt_limits[2 * index]:pa.cnt_limits[2 * index + 1]]

            # Plot the bonded particles in black
            ax.scatter(pa.x[bond_indices_i], pa.y[bond_indices_i], pa.z[bond_indices_i], c='red', label='Bonded particles')

            # Highlight the current particle in red
            ax.scatter(x_val, y_val, z_val, c='black', label=f'Particle {index}')

            # Set titles and labels
            ax.set_title(f"Particle {index} (3D)")
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.set_zlabel("Z coordinate")
            ax.legend()

        elif dim == 2:
            # Plotting in 2D
            ax = fig.add_subplot(111)

            # Plot all particles in blue
            ax.scatter(pa.x, pa.y, c='blue', alpha=0.5, label="All particles")

            # Get the bonded particles for the current particle
            bond_indices_i = pa.cnt_idxs[pa.cnt_limits[2 * index]:pa.cnt_limits[2 * index + 1]]

            # Plot the bonded particles in black
            ax.scatter(pa.x[bond_indices_i], pa.y[bond_indices_i], c='black', label='Bonded particles')

            # Highlight the current particle in red
            ax.scatter(x_val, y_val, c='red', label=f'Particle {index}')

            # Set titles and labels
            ax.set_title(f"Particle {index} (2D)")
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.legend()

        # Show the plot if requested (this will open the plot in an interactive window)
        if show_plot:
            plt.show()

        # Save the figure with a unique name inside the specified folder
        plt.savefig(f"{folder_name}/particle_{index}_dim{dim}.png")

        # Close the plot to avoid overlapping with the next plot
        plt.close()

    # print(f"Figures saved in folder: {folder_name}")


def add_properties_stride(pa, stride=1, *props):
    for prop in props:
        pa.add_property(name=prop, stride=stride)


def get_particle_array_peridynamics(constants=None, dim=2, **props):
    pd_props = ['x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rad', 'fx', 'fy', 'fz', 'is_static',
                'fx_imposed',
                'fy_imposed',
                'fz_imposed',
                'u_imposed',
                'v_imposed',
                'w_imposed',
                'is_dynamic']
    consts = {
        'no_bonds_limits': np.array([30], dtype='int'),
        'criterion_dist': -1.
    }

    if constants:
        consts.update(constants)
    pa = get_particle_array(constants=consts,
                            additional_props=pd_props, **props)

    # contact indices
    pa.add_property('cnt_idxs', stride=consts['no_bonds_limits'], type='int')
    # each particle contact limits
    pa.add_property('cnt_limits', stride=2, type='int')
    # each particle total number of contacts
    pa.add_property('tot_cnts', type='int')

    # initial bond length
    pa.add_property('undeformed_bond_length', stride=consts['no_bonds_limits'])
    pa.add_property('deformed_bond_length', stride=consts['no_bonds_limits'])
    pa.add_property('bond_damage', stride=consts['no_bonds_limits'])
    pa.add_property('bond_pd_fx', stride=consts['no_bonds_limits'])
    pa.add_property('bond_pd_fy', stride=consts['no_bonds_limits'])
    pa.add_property('bond_pd_fz', stride=consts['no_bonds_limits'])

    # set the contacts to default values (this is general to all contact
    # tracking algorithms)
    pa.cnt_idxs[:] = -1
    pa.cnt_limits[:] = 0
    pa.tot_cnts[:] = 0

    # initialize peridynamcis specific variables
    pa.bond_damage[:] = 1.
    pa.bond_pd_fx[:] = 0.
    pa.bond_pd_fy[:] = 0.
    pa.bond_pd_fz[:] = 0.

    set_contacts_pd(pa, pa.criterion_dist[0], dim)

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p',
        'cnt_idxs',
        'cnt_limits',
        'tot_cnts',
        'undeformed_bond_length',
        'deformed_bond_length',
        'bond_damage',
        'bond_pd_fx',
        'bond_pd_fy',
        'bond_pd_fz'
    ])

    return pa


class SetContactsPD(Equation):
    def __init__(self, dest, sources, criterion_dist=0.3):
        super(SetContactsPD, self).__init__(dest, sources)
        self.criterion_dist = criterion_dist

    def loop(self, d_idx, d_x, d_y, d_z, d_cnt_idxs, d_cnt_limits,
             d_tot_cnts, d_no_bonds_limits, d_undeformed_bond_length,
             s_idx, s_x, s_y, s_z, RIJ):
        i = declare('int')
        if d_idx != s_idx:
            if RIJ < self.criterion_dist:
                # add the contact index at the end of the list
                i = d_idx * d_no_bonds_limits[0] + d_tot_cnts[d_idx]
                d_cnt_idxs[i] = s_idx
                d_undeformed_bond_length[i] = RIJ

                # increment the total number of contacts
                d_tot_cnts[d_idx] += 1

                # set the no of bonds limit
                d_cnt_limits[2 * d_idx] = d_idx * d_no_bonds_limits[0]
                d_cnt_limits[2 * d_idx + 1] = (
                    d_idx * d_no_bonds_limits[0] + d_tot_cnts[d_idx])


def set_contacts_pd(pa, criterion_dist, dim):
    assert criterion_dist > 0., "the criterion_dist has to be positive"
    assert pa.no_bonds_limits > 0, "Number of max bonds needs to be positive"

    equations = [
        Group(
            equations=[SetContactsPD(dest=pa.name, sources=[pa.name])])
    ]

    sph_eval = SPHEvaluator(arrays=[pa], equations=equations, dim=dim,
                            kernel=QuinticSpline(dim=dim))

    sph_eval.evaluate(0.1, 0.1)


class PDGTVFIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.stage1()
        self.do_post_stage(dt, 1)

        self.compute_accelerations(0, update_nnps=False)

        self.stage2()
        # We update domain here alone as positions only change here.
        # self.update_domain()
        self.do_post_stage(dt, 2)

        self.compute_accelerations(1)

        self.stage3()
        self.do_post_stage(dt, 3)


class GTVFStepPeridynamics(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
               d_fx, d_fy, d_fz, d_rho, dt):
        dtb2 = dt / 2.

        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_fx[d_idx] / d_rho[d_idx]
        d_v[d_idx] += dtb2 * d_fy[d_idx] / d_rho[d_idx]
        d_w[d_idx] += dtb2 * d_fz[d_idx] / d_rho[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
               dt):
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]

    def stage3(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
               d_fx, d_fy, d_fz, d_rho, dt):
        dtb2 = dt / 2.

        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_fx[d_idx] / d_rho[d_idx]
        d_v[d_idx] += dtb2 * d_fy[d_idx] / d_rho[d_idx]
        d_w[d_idx] += dtb2 * d_fz[d_idx] / d_rho[d_idx]


class ResetForce(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.


class ApplyStaticAndDynamicBC(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz,
                   d_u, d_v, d_w,
                   d_is_static,
                   d_fx_imposed,
                   d_fy_imposed,
                   d_fz_imposed,
                   d_u_imposed,
                   d_v_imposed,
                   d_w_imposed,
                   d_is_dynamic):
        if d_is_static[d_idx] == 1.:
            d_fx[d_idx] = 0.
            d_fy[d_idx] = 0.
            d_fz[d_idx] = 0.

            d_u[d_idx] = 0.
            d_v[d_idx] = 0.
            d_w[d_idx] = 0.

        if d_is_dynamic[d_idx] == 1.:
            d_fx[d_idx] += d_fx_imposed[d_idx]
            d_fy[d_idx] += d_fy_imposed[d_idx]
            d_fz[d_idx] += d_fz_imposed[d_idx]

            d_u[d_idx] = d_u_imposed[d_idx]
            d_v[d_idx] = d_v_imposed[d_idx]
            d_w[d_idx] = d_w_imposed[d_idx]


class BondBasedElasticPDForce(Equation):
    def initialize(self, d_idx, d_x, d_y, d_z,
                   d_u, d_v, d_w,
                   d_cnt_idxs, d_cnt_limits,
                   d_tot_cnts,
                   d_bond_damage,
                   d_deformed_bond_length,
                   d_undeformed_bond_length,
                   d_bond_pd_fx,
                   d_bond_pd_fy,
                   d_bond_pd_fz,
                   d_fx, d_fy, d_fz,
                   d_m, d_c, dt):
        i, p, q, sidx = declare('int', 4)
        # particle d_idx has its neighbours information in d_cnt_idxs
        # The range of such is
        p = d_cnt_limits[2 * d_idx]
        q = d_cnt_limits[2 * d_idx + 1]

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

            # =========================
            # Compute the bond force
            # =========================
            # deformed bond length
            d_deformed_bond_length[i] = rij
            # bond force direction
            bond_direction_x = n_ij_x
            bond_direction_y = n_ij_y
            bond_direction_z = n_ij_z

            # bond force
            s = (d_deformed_bond_length[i] - d_undeformed_bond_length[i]) / d_undeformed_bond_length[i]
            tmp = d_c[0] * d_bond_damage[i] * s
            # tmp = d_c[0] * s
            d_bond_pd_fx[i] = tmp * bond_direction_x
            d_bond_pd_fy[i] = tmp * bond_direction_y
            d_bond_pd_fz[i] = tmp * bond_direction_z
            # ============================
            # Compute the bond force ends
            # ============================

            # add the force to the global force of particle i
            d_fx[d_idx] += d_bond_pd_fx[i]
            d_fy[d_idx] += d_bond_pd_fy[i]
            d_fz[d_idx] += d_bond_pd_fz[i]


class PeridynamicsScheme(Scheme):
    def __init__(self, solids, dim, gx=0., gy=0., gz=0.):
        self.solids = solids

        self.dim = dim

        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

        self.attributes_changed()

    # def add_user_options(self, group):
    #     group.add_argument("--kr-stiffness", action="store",
    #                        dest="kr", default=1e8,
    #                        type=float,
    #                        help="Repulsive spring stiffness")

    #     group.add_argument("--kf-stiffness", action="store",
    #                        dest="kf", default=1e3,
    #                        type=float,
    #                        help="Tangential spring stiffness")

    #     group.add_argument("--fric-coeff", action="store",
    #                        dest="fric_coeff", default=0.0,
    #                        type=float,
    #                        help="Friction coefficient")

    # def consume_user_options(self, options):
    #     _vars = ['kr', 'kf', 'fric_coeff']
    #     data = dict((var, self._smart_getattr(options, var)) for var in _vars)
    #     self.configure(**data)

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        bodystep = GTVFStepPeridynamics()
        integrator_cls = PDGTVFIntegrator

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
                BondBasedElasticPDForce(
                    dest=name,
                    sources=None))

            g5.append(
                ApplyStaticAndDynamicBC(
                    dest=name,
                    sources=None))

        stage2.append(Group(equations=g5, real=False))

        return MultiStageEquations([stage1, stage2])

    def get_solver(self):
        return self.solver
