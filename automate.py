#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import h5py

from itertools import cycle, product
import json
from automan.api import Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores
# from pysph.solver.utils import load, get_files
from automan.api import (Automator, Simulation, filter_cases, filter_by_name)
from automan.automation import (CommandTask)

import numpy as np
import matplotlib
matplotlib.use('agg')
from cycler import cycler
from matplotlib import rc, patches, colors
from matplotlib.collections import PatchCollection

rc('font', **{'family': 'Helvetica', 'size': 12})
rc('legend', fontsize='medium')
rc('axes', grid=True, linewidth=1.2)
rc('axes.grid', which='both', axis='both')
# rc('axes.formatter', limits=(1, 2), use_mathtext=True, min_exponent=1)
rc('grid', linewidth=0.5, linestyle='--')
rc('xtick', direction='in', top=True)
rc('ytick', direction='in', right=True)
rc('savefig', format='pdf', bbox='tight', pad_inches=0.05,
   transparent=False, dpi=300)
rc('lines', linewidth=1.5)
rc('axes', prop_cycle=(
    cycler('color', ['tab:blue', 'tab:green', 'tab:red',
                     'tab:orange', 'm', 'tab:purple',
                     'tab:pink', 'tab:gray']) +
    cycler('linestyle', ['-.', '--', '-', ':',
                         (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                         (0, (3, 2, 1, 1)), (0, (3, 2, 2, 1, 1, 1)),
                         ])
))


# n_core = 6
n_core = 16
n_thread = n_core * 2
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params

def get_files(directory):
    # =====================================
    # start: get the files and sort
    # =====================================
    files = [filename for filename in os.listdir(directory) if filename.startswith("particles") and filename.endswith("h5") ]
    files.sort()
    files_num = []
    for f in files:
        f_last = f[10:]
        files_num.append(int(f_last[:-3]))
    files_num.sort()

    sorted_files = []
    for num in files_num:
        sorted_files.append("particles_" + str(num) + ".h5")
    files = sorted_files
    return files


class Test01ElasticNormalImpactOfTwoIdenticalParticles(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'test01_elastic_normal_impact_of_two_identical_particles'

    def setup(self):
        get_path = self.input_path

        cmd = './build/examples/01ElasticNormalImpactOfTwoIdenticalParticles ./examples/inputs/01_elastic_normal_impact_of_two_identical_particles.json $output_dir'
        # Base case info
        self.case_info = {
            'case_1': (dict(
                ), 'Cabana'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_time_vs_normal_force()

    def plot_time_vs_normal_force(self):
        data = {}
        for name in self.case_info:
            files = get_files(self.input_path(name))

            # print(directory_name+files[0])
            frc = []
            time = []
            for f in files:
                f = h5py.File(self.input_path(name, f), "r")
                frc.append(f["forces"][1][0])
                time.append(f.attrs["Time"])

            plt.plot(time, frc, "^-", label="Cabana DEM solver")
            plt.legend()
            plt.savefig(self.input_path(name, "force_fn_vs_time.pdf"))


class Test02ElasticNormalImpactParticleWall(Problem):
    """

    """
    def get_name(self):
        return 'test02_elastic_normal_impact_of_particle_wall'

    def setup(self):
        get_path = self.input_path

        cmd = './build/examples/02ElasticNormalImpactOfParticleWall ./examples/inputs/02_elastic_normal_impact_of_particle_wall.json $output_dir'
        # Base case info
        self.case_info = {
            'case_1': (dict(
                ), 'Cabana'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_time_vs_normal_force()

    def plot_time_vs_normal_force(self):
        data = {}
        for name in self.case_info:
            files = get_files(self.input_path(name))

            # print(directory_name+files[0])
            frc = []
            time = []
            for f in files:
                f = h5py.File(self.input_path(name, f), "r")
                frc.append(f["forces"][0][1])
                time.append(f.attrs["Time"])

            plt.plot(time, frc, "^-", label="Cabana DEM solver")
            plt.legend()
            plt.savefig(self.input_path(name, "force_fn_vs_time.pdf"))


class Test03NormalParticleWallDifferentCOR(Problem):
    """

    """
    def get_name(self):
        return 'test03_normal_particle_wall_different_cor'

    def setup(self):
        get_path = self.input_path

        cmd = './build/examples/03NormalParticleWallDifferentCOR ./examples/inputs/03_normal_particle_wall_different_cor.json $output_dir'
        # Base case info
        self.case_info = {
            'cor_0_01': (dict(
                cor_pw=0.01,
                ), 'COR=0.01'),

            'cor_0_2': (dict(
                cor_pw=0.2,
                ), 'COR=0.2'),

            'cor_0_4': (dict(
                cor_pw=0.4,
                ), 'COR=0.4'),

            'cor_0_6': (dict(
                cor_pw=0.6,
                ), 'COR=0.6'),

            'cor_0_8': (dict(
                cor_pw=0.8,
                ), 'COR=0.8'),

            'cor_1_0': (dict(
                cor_pw=1.0,
                ), 'COR=1.0'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_cor_input_vs_output()

    def plot_cor_input_vs_output(self):
        cor_input_analytical = np.linspace(0., 1., 20)
        cor_output_analytical = np.linspace(0., 1., 20)
        input_velocity = 3.9

        cor_input_simulation = []
        cor_output_simulation = []
        for name in self.case_info:
            # check if the files exist then run this postprocess
            files = get_files(self.input_path(name))

            if len(files) > 0:
                f = h5py.File(self.input_path(name, files[-1]), "r")
                cor_input_i = self.case_info[name][0]["cor_pw"]
                cor_output_i = f["velocities"][0][1] / input_velocity
                # TODO: Before plotting save the extracted data in an npz file
                plt.plot(cor_input_analytical, cor_output_analytical, label="Analytical")
                plt.scatter([cor_input_i], [cor_output_i], label="Cabana DEM solver")
                plt.legend()
                plt.savefig(self.input_path(name, "cor_input_vs_output.pdf"))
                plt.clf()

                cor_input_simulation.append(cor_input_i)
                cor_output_simulation.append(cor_output_i)

        # Load specific extracted data from the output folders, which is saved
        # in npz files, to plot the comparision between all the existing files
        plt.clf()
        plt.plot(cor_input_analytical, cor_output_analytical, label="Analytical")
        plt.scatter(cor_input_simulation, cor_output_simulation, label="Cabana DEM solver")
        plt.legend()
        path_to_figure, _tail = os.path.split(name)
        plt.savefig(self.input_path(path_to_figure, "cor_input_vs_output.pdf"))


class Test04ElasticNormalImpactParticleWall(Problem):
    """

    """
    def get_name(self):
        return 'test04_oblique_particle_wall_different_angles'

    def setup(self):
        get_path = self.input_path

        cmd = './build/examples/04ObliqueParticleWallDifferentAngles ./examples/inputs/04_oblique_particle_wall_different_angles.json $output_dir'
        # Base case info
        self.case_info = {
            'angle_2': (dict(
                angle=2.,
                ), 'Angle=2.'),

            'angle_5': (dict(
                angle=5.,
                ), 'Angle=5.'),

            'angle_10': (dict(
                angle=10.,
                ), 'Angle=10.'),

            'angle_15': (dict(
                angle=15.,
                ), 'Angle=15.'),

            'angle_20': (dict(
                angle=20.,
                ), 'Angle=20.'),

            'angle_25': (dict(
                angle=25.,
                ), 'Angle=25.'),

            'angle_30': (dict(
                angle=30.,
                ), 'Angle=30.'),

            'angle_35': (dict(
                angle=35.,
                ), 'Angle=35.'),

            'angle_40': (dict(
                angle=40.,
                ), 'Angle=40.'),

            'angle_50': (dict(
                angle=50.,
                ), 'Angle=50.'),

            'angle_55': (dict(
                angle=55.,
                ), 'Angle=55.'),

            'angle_60': (dict(
                angle=60.,
                ), 'Angle=60.'),

            'angle_65': (dict(
                angle=65.,
                ), 'Angle=65.'),

            'angle_70': (dict(
                angle=70.,
                ), 'Angle=70.'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_time_vs_normal_force()

    def plot_time_vs_normal_force(self):
        data_incident_angle_vs_omega_exp = np.loadtxt(os.path.join(
            "examples/validation_data",
            '04_oblique_particle_wall_different_angles_omega_vs_incident_angle_Kharaz_Gorham_Salman_experimental_data.csv'),
                                                      delimiter=',')
        incident_angle_exp, omega_exp = data_incident_angle_vs_omega_exp[:, 0], data_incident_angle_vs_omega_exp[:, 1]

        data_incident_angle_vs_omega_Lethe_DEM = np.loadtxt(os.path.join(
            "examples/validation_data",
            '04_oblique_particle_wall_different_angles_omega_vs_incident_angle_Lethe_DEM_data.csv'),
                                                      delimiter=',')
        incident_angle_Lethe_DEM, omega_Lethe_DEM = data_incident_angle_vs_omega_Lethe_DEM[:, 0], data_incident_angle_vs_omega_Lethe_DEM[:, 1]

        incident_angle_simulation = []
        omega_simulation = []
        for name in self.case_info:
            # check if the files exist then run this postprocess
            files = get_files(self.input_path(name))

            if len(files) > 0:
                f = h5py.File(self.input_path(name, files[-1]), "r")
                angle_i = self.case_info[name][0]["angle"]
                omega_i = f["omega"][0][2]
                # TODO: Before plotting save the extracted data in an npz file
                plt.plot(incident_angle_exp, omega_exp, "*-", label="Experiment")
                plt.plot(incident_angle_Lethe_DEM, omega_Lethe_DEM, "v--", label="Lethe DEM solver")
                plt.scatter([angle_i], [omega_i], label="Cabana DEM solver")
                plt.legend()
                plt.savefig(self.input_path(name, "incident_angle_vs_omega.pdf"))
                plt.clf()

                incident_angle_simulation.append(angle_i)
                omega_simulation.append(omega_i)

        # Load specific extracted data from the output folders, which is saved
        # in npz files, to plot the comparision between all the existing files
        plt.clf()
        # print(incident_angle_exp, omega_exp)
        plt.scatter(incident_angle_exp, omega_exp, label="Experiment")
        plt.plot(incident_angle_Lethe_DEM, omega_Lethe_DEM, "v--", label="Lethe DEM solver")
        plt.plot(incident_angle_simulation, omega_simulation, "^-", label="Cabana DEM solver")
        plt.legend()
        path_to_figure, tail = os.path.split(name)
        plt.savefig(self.input_path(path_to_figure, "incident_angle_vs_omega.pdf"))


if __name__ == '__main__':
    PROBLEMS = [
        # Image generator
        Test01ElasticNormalImpactOfTwoIdenticalParticles,
        Test02ElasticNormalImpactParticleWall,
        Test03NormalParticleWallDifferentCOR,
        Test04ElasticNormalImpactParticleWall
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
