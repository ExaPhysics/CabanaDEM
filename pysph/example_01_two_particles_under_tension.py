from __future__ import print_function
import numpy as np
import os

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator

from pysph.sph.equation import Group
from pysph.solver.application import Application
from bonded_dem import (get_particle_array_bonded_dem,
                        SpringBondModel)
from pysph.sph.scheme import SchemeChooser


class TwoParticlesUnderTension(Application):
    def initialize(self):
        self.length = 1.
        self.height = 0.1

    def create_particles(self):
        # each particle has a radius of rad
        rad = 0.1
        rho = 2000.
        m = 4. / 3. * np.pi * rad**3. * rho
        moi = 2. / 5. * np.pi * rad**2.
        # x = np.arange(0., 1., 2. * rad)
        x = np.array([0., 2. * rad])
        y = np.zeros(len(x))
        beam = get_particle_array(
            m=m,
            moi=moi,
            x=x,
            y=y,
            h=3. * rad,
            rad=rad,
            name="beam",
            constants=dict(
                no_bonds_limits=1,
                criterion_dist=3. * rad)
            )
        self.scheme.setup_properties([beam])
        beam.u[1] = 10.0

        # set the static particle indices

        indices = []
        min_x = min(beam.x)
        for i in range(len(beam.y)):
            if beam.x[i] < min_x + 1. * rad:
                indices.append(i)

        # add static particle data
        beam.is_static[indices] = 1.

        return [beam]

    def configure_scheme(self):
        dt = 1e-4
        tf = 1.
        pfreq = 300

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=pfreq)

    def create_scheme(self):
        spring_bond_model = SpringBondModel(
            solids=['beam'],
            dim=2)
        s = SchemeChooser(default='spring', spring=spring_bond_model)
        return s

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files

        info = self.read_info(fname)
        output_files = self.output_files

        from pysph.solver.utils import iter_output

        t = []
        x = []

        for sd, rb in iter_output(output_files, 'beam'):
            _t = sd['t']
            t.append(_t)
            x.append(rb.x[-1])

        plt.clf()
        plt.plot(t, x, '-', label='x vs t')

        plt.title('x_vs_t')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "t_vs_x.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================


if __name__ == '__main__':
    app = TwoParticlesUnderTension()
    app.run()
    app.post_process(app.info_filename)
