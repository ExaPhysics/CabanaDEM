"""A sphere of density 500 falling into a hydrostatic tank (15 minutes)

Check basic equations of SPH to throw a ball inside the vessel
"""
from __future__ import print_function
import numpy as np

from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator

from pysph.sph.equation import Group
from pysph.solver.application import Application
from bonded_dem import (get_particle_array_bonded_dem,
                        SpringBondModel)
from pysph.sph.scheme import SchemeChooser


def test_3_x_3_bonded_particles_contacts():
    """
    x    x    x



    x    x    x



    x    x    x

    """
    rad = 0.1
    # create 3 rows and 3 columns of particles
    x = np.array(
        [-2. * rad, 0., 2. * rad, -2. * rad, 0., 2. * rad, -2. * rad, 0.,
         2. * rad])
    y = np.array(
        [-2. * rad, -2. * rad, -2. * rad, 0., 0., 0., 2. * rad, 2. * rad,
         2. * rad])

    z = np.array(
        [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    beam = get_particle_array_bonded_dem(
        x=x,
        y=y,
        z=z,
        h=3. * rad,
        rad=rad,
        name="beam",
        constants=dict(
            no_bonds_limits=8,
            criterion_dist=3. * rad)
    )
    # print(beam.cnt_idxs.reshape(9, 8))
    # print(beam.cnt_idxs[0:8])
    # return [beam]


if __name__ == '__main__':
    test_3_x_3_bonded_particles_contacts()
