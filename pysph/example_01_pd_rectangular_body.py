"""
https://kaipartmann.github.io/Peridynamics.jl/stable/generated/tutorial_tension_static/
"""
from __future__ import print_function
import numpy as np
import os

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator

from pysph.sph.equation import Group
from pysph.solver.application import Application
from peridynamics import (get_particle_array_peridynamics,
                          PeridynamicsScheme,
                          scatter_bonded_particles)

from pysph.sph.scheme import SchemeChooser
from peridynamics import PeridynamicsScheme


class PDTensileTestQuasiStatic(Application):
    def initialize(self):
        self.dim = 2
        self.length = 1.
        self.height = 1.
        self.spacing = 0.1

    def create_particles(self):
        import pysph.tools.geometry as G
        x, y = G.get_2d_block(self.spacing, self.length, self.height)
        # each particle has a radius of rad
        spacing = self.spacing
        body = get_particle_array_peridynamics(
            dim=self.dim,
            m=1.,
            x=x,
            y=y,
            z=0.,
            h=1. * spacing,
            name="body",
            constants=dict(
                no_bonds_limits=30,
                criterion_dist=3. * spacing)
            )
        # self.scheme.setup_properties([body])
        # body.u[-1] = -10.0

        # # set the velocities
        # indices = []
        # min_x = min(body.x)
        # for i in range(len(body.y)):
        #     if body.x[i] < min_x + 0.2 * spacing:
        #         indices.append(i)

        # # add static particle data
        # body.is_static[indices] = 1.

        # Mark the indices with dynamic boundary condition
        # set the velocities
        indices_left = []
        min_x = min(body.x)
        for i in range(len(body.y)):
            if body.x[i] < min_x + 0.2 * spacing:
                indices_left.append(i)

        # add static particle data
        u_imposed = np.zeros(len(body.x))
        u_imposed[indices_left] = -10.

        indices_right = []
        max_x = max(body.x)
        for i in range(len(body.y)):
            if body.x[i] > max_x - 0.2 * spacing:
                indices_right.append(i)
        u_imposed[indices_right] = 10.

        body.is_dynamic[indices_left] = 1.
        body.is_dynamic[indices_right] = 1.

        body.u_imposed[:] = u_imposed[:]

        # scatter_bonded_particles(body, indices, "example_01_pd_rectangular_body_output/bonded_particles_data_figures_output", 2, False)
        return [body]

    def configure_scheme(self):
        dt = 1e-4
        # tf = 1.
        tf = 1000. * dt
        pfreq = 100

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=pfreq)

    def create_scheme(self):
        pd_model = PeridynamicsScheme(
            solids=['body'],
            dim=3)
        s = SchemeChooser(default='pd', pd=pd_model)
        return s


if __name__ == '__main__':
    app = PDTensileTestQuasiStatic()
    app.run()
    app.post_process(app.info_filename)
