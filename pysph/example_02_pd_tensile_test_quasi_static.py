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


class PDTensileTestQuasiStatic(Application):
    def initialize(self):
        self.dim = 3
        self.length = 1.
        self.height = 0.1
        self.spacing = 9.96 * 1e-4

    def create_particles(self):
        import meshio

        mesh = meshio.read('TensileTestMesh.inp')
        # Get node coordinates
        nodes = mesh.points

        # Loop over each cell block (for hexahedrons)
        x_centroid = []
        y_centroid = []
        z_centroid = []
        for cell_block in mesh.cells:
            if cell_block.type == "hexahedron":
                # Loop over each cell's connectivity
                for cell in cell_block.data:
                    # Get the coordinates of the 8 nodes for this hexahedron
                    cell_coords = nodes[cell]

                    # Compute the centroid by averaging the coordinates
                    centroid = np.mean(cell_coords, axis=0)
                    x_centroid.append(centroid[0])
                    y_centroid.append(centroid[1])
                    z_centroid.append(centroid[2])
                    # print(f"Centroid of the hexahedron: {centroid}")

        # each particle has a radius of rad
        spacing = self.spacing
        print("Spacing we use", spacing)
        print("Compare spacing")
        print(((x_centroid[7468] - x_centroid[7469])**2. + (y_centroid[7468] - y_centroid[7469])**2. + (z_centroid[7468] - z_centroid[7469])**2.)**0.5)
        body = get_particle_array_peridynamics(
            dim=self.dim,
            m=1.,
            x=x_centroid,
            y=y_centroid,
            z=z_centroid,
            h=1. * spacing,
            name="body",
            constants=dict(
                no_bonds_limits=300,
                criterion_dist=3. * spacing)
            )
        # self.scheme.setup_properties([body])
        # body.u[-1] = -10.0

        # set the static particle indices

        indices = []
        min_x = min(body.x)
        for i in range(len(body.y)):
            if body.x[i] < min_x + 1. * spacing:
                indices.append(i)

        # add static particle data
        body.is_static[indices] = 1.
        scatter_bonded_particles(body, [1], "example_02_pd_tensile_test_quasi_static_output/bonded_particles_data_figures_output", dim=self.dim, show_plot=False)

        return [body]

    def configure_scheme(self):
        dt = 1e-4
        # tf = 1.
        tf = 300. * dt
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
