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

rc('font', **{'family': 'sans-serif', 'size': 12})
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
        self.move_figures()

    def plot_time_vs_normal_force(self):
        # data_incident_overlap_vs_fn_analy = np.loadtxt(os.path.join(
        #     "examples/validation_data",
        #     '01_elastic_normal_impact_of_two_identical_particles_overlap_vs_fn_analytical_data.csv'),
        #                                               delimiter=',')
        # overlap_analy, fn_analy = data_incident_overlap_vs_fn_analy[:, 0], data_incident_overlap_vs_fn_analy[:, 1]

        data_incident_time_vs_fn_analy = np.loadtxt(os.path.join(
            "examples/validation_data",
            '01_elastic_normal_impact_of_two_identical_particles_time_vs_fn_analytical_data.csv'),
                                                      delimiter=',')
        time_analy, fn_analy = data_incident_time_vs_fn_analy[:, 0], data_incident_time_vs_fn_analy[:, 1]

        for name in self.case_info:
            files = get_files(self.input_path(name))

            # TODO add one more condition to skip this plot and go to direct comparision plots
            if len(files) > 0:
                # print(directory_name+files[0])
                fn_simu = []
                time_simu = []
                for f in files:
                    f = h5py.File(self.input_path(name, f), "r")
                    fn_simu.append(f["forces"][1][0] / 1e3)
                    time_simu.append(f.attrs["Time"] / 1e-6)

                # save the simulated data in a npz file in the case folder
                res_npz = os.path.join(self.input_path(name, "results.npz"))
                np.savez(res_npz,
                         time_simu=time_simu,
                         fn_simu=fn_simu)

                plt.scatter(time_analy, fn_analy, label="Analytical")
                plt.plot(time_simu, fn_simu, "^-", label="Cabana DEM solver")
                plt.legend()
                plt.savefig(self.input_path(name, "fn_vs_time.pdf"))

        plt.clf()
        plt.scatter(time_analy, fn_analy, label="Analytical")
        for name in self.case_info:
            # save the simulated data in a npz file in the case folder
            res_npz = np.load(os.path.join(self.input_path(name, "results.npz")))
            plt.plot(res_npz["time_simu"], res_npz["fn_simu"], "^-", label="Cabana DEM solver " + self.case_info[name][1])

        plt.legend()
        path_to_figure, tail = os.path.split(name)
        plt.savefig(self.input_path(path_to_figure, "fn_vs_time.pdf"))

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            # source =
            source, tail = os.path.split(self.input_path(name))
            # print("full source", source)
            # print(source[8:], "source 8 is")
            target_dir = "manuscript/figures/" + source[8:] + "/"
            try:
                os.makedirs(target_dir)
            except FileExistsError:
                pass

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Test02ElasticNormalImpactParticleWall(Test01ElasticNormalImpactOfTwoIdenticalParticles):
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
        self.move_figures()

    def plot_time_vs_normal_force(self):
        data_incident_time_vs_fn_analy = np.loadtxt(os.path.join(
            "examples/validation_data",
            '01_elastic_normal_impact_of_two_identical_particles_time_vs_fn_analytical_data.csv'),
                                                      delimiter=',')
        time_analy, fn_analy = data_incident_time_vs_fn_analy[:, 0], data_incident_time_vs_fn_analy[:, 1]

        for name in self.case_info:
            files = get_files(self.input_path(name))

            # TODO add one more condition to skip this plot and go to direct comparision plots
            if len(files) > 0:
                # print(directory_name+files[0])
                fn_simu = []
                time_simu = []
                for f in files:
                    f = h5py.File(self.input_path(name, f), "r")
                    fn_simu.append(f["forces"][0][1] / 1e3)
                    time_simu.append(f.attrs["Time"] / 1e-6)

                # save the simulated data in a npz file in the case folder
                res_npz = os.path.join(self.input_path(name, "results.npz"))
                np.savez(res_npz,
                         time_simu=time_simu,
                         fn_simu=fn_simu)

                plt.scatter(time_analy, fn_analy, label="Analytical")
                plt.plot(time_simu, fn_simu, "^-", label="Cabana DEM solver")
                plt.legend()
                plt.savefig(self.input_path(name, "fn_vs_time.pdf"))

        plt.clf()
        plt.scatter(time_analy, fn_analy, label="Analytical")
        for name in self.case_info:
            # save the simulated data in a npz file in the case folder
            res_npz = np.load(os.path.join(self.input_path(name, "results.npz")))
            plt.plot(res_npz["time_simu"], res_npz["fn_simu"], "^-", label="Cabana DEM solver " + self.case_info[name][1])

        plt.legend()
        path_to_figure, tail = os.path.split(name)
        plt.savefig(self.input_path(path_to_figure, "fn_vs_time.pdf"))


class Test03NormalParticleWallDifferentCOR(Test01ElasticNormalImpactOfTwoIdenticalParticles):
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
        self.move_figures()

    def plot_cor_input_vs_output(self):
        cor_input_analytical = np.linspace(0., 1., 20)
        cor_output_analytical = np.linspace(0., 1., 20)
        input_velocity = 3.9

        for name in self.case_info:
            # check if the files exist then run this postprocess
            files = get_files(self.input_path(name))

            if len(files) > 0:
                f = h5py.File(self.input_path(name, files[-1]), "r")
                cor_input_i = self.case_info[name][0]["cor_pw"]
                cor_output_i = f["velocities"][0][1] / input_velocity
                # save the simulated data in a npz file in the case folder
                res_npz = os.path.join(self.input_path(name, "results.npz"))
                np.savez(res_npz,
                         cor_input_i=[cor_input_i],
                         cor_output_i=[cor_output_i])

                plt.plot(cor_input_analytical, cor_output_analytical, label="Analytical")
                plt.scatter([cor_input_i], [cor_output_i], label="Cabana DEM solver")
                plt.legend()
                plt.savefig(self.input_path(name, "cor_input_vs_output.pdf"))
                plt.clf()

        # Load specific extracted data from the output folders, which is saved
        # in npz files, to plot the comparision between all the existing files
        plt.clf()
        plt.plot(cor_input_analytical, cor_output_analytical, label="Analytical")
        cor_input_simulation = []
        cor_output_simulation = []
        for name in self.case_info:
            # save the simulated data in a npz file in the case folder
            res_npz = np.load(os.path.join(self.input_path(name, "results.npz")))
            cor_input_simulation.append(res_npz["cor_input_i"])
            cor_output_simulation.append(res_npz["cor_output_i"])

        plt.scatter(cor_input_simulation, cor_output_simulation, label="Cabana DEM solver")
        plt.legend()
        path_to_figure, _tail = os.path.split(name)
        plt.savefig(self.input_path(path_to_figure, "cor_input_vs_output.pdf"))


class Test04ElasticNormalImpactParticleWall(Test01ElasticNormalImpactOfTwoIdenticalParticles):
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
        self.move_figures()

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

        for name in self.case_info:
            # check if the files exist then run this postprocess
            files = get_files(self.input_path(name))

            if len(files) > 0:
                f = h5py.File(self.input_path(name, files[-1]), "r")
                angle_i = self.case_info[name][0]["angle"]
                omega_i = f["omega"][0][2]
                # save the simulated data in a npz file in the case folder
                res_npz = os.path.join(self.input_path(name, "results.npz"))
                np.savez(res_npz,
                         angle_i=[angle_i],
                         omega_i=[omega_i])

                plt.plot(incident_angle_exp, omega_exp, "*-", label="Experiment")
                plt.plot(incident_angle_Lethe_DEM, omega_Lethe_DEM, "v--", label="Lethe DEM solver")
                plt.scatter([angle_i], [omega_i], label="Cabana DEM solver")
                plt.legend()
                plt.savefig(self.input_path(name, "incident_angle_vs_omega.pdf"))
                plt.clf()

        # Load specific extracted data from the output folders, which is saved
        # in npz files, to plot the comparision between all the existing files
        plt.clf()
        plt.scatter(incident_angle_exp, omega_exp, label="Experiment")
        plt.plot(incident_angle_Lethe_DEM, omega_Lethe_DEM, "v--", label="Lethe DEM solver")

        incident_angle_simulation = []
        omega_simulation = []
        for name in self.case_info:
            # save the simulated data in a npz file in the case folder
            res_npz = np.load(os.path.join(self.input_path(name, "results.npz")))
            incident_angle_simulation.append(res_npz["angle_i"])
            omega_simulation.append(res_npz["omega_i"])

        plt.plot(incident_angle_simulation, omega_simulation, "^-", label="Cabana DEM solver")
        plt.legend()
        path_to_figure, tail = os.path.split(name)
        plt.savefig(self.input_path(path_to_figure, "incident_angle_vs_omega.pdf"))


class Tst03MultipleParticlesContacts(Problem):
    """
    To test the tangential contact, and tracking the indices during a DEM simulation.
    A single particle hits two particles, in the process we check if the implemented
    DEM algorithm is behaving as expected.
    """
    def get_name(self):
        return 'tst03_multiple_particles_contact'

    def setup(self):
        get_path = self.input_path

        cmd = './build/examples/Tst03MultipleParticlesContacts ./examples/inputs/tst_03_multiple_particles_contacts.json $output_dir'
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
        self.move_figures()

    def plot_time_vs_normal_force(self):
        for name in self.case_info:
            files = get_files(self.input_path(name))

            if len(files) > 0:
                # get total no of particles
                f = h5py.File(self.input_path(name, files[0]), "r")
                no_of_particles = len(f["forces"])

                # create an array to track the total no of contacts with time
                # for all the particles
                total_contacts_simu_list = []
                for i in range(no_of_particles):
                    total_contacts_simu_list.append([])
                total_contacts_simu = total_contacts_simu_list

                time_simu = []
                for f in files:
                    f = h5py.File(self.input_path(name, f), "r")

                    total_no_tangential_contacts = f["total_no_tangential_contacts"]
                    for i in range(no_of_particles):
                        total_contacts_simu[i].append(total_no_tangential_contacts[i])
                    time_simu.append(f.attrs["Time"] / 1e-6)

                for i in range(no_of_particles):
                    plt.plot(time_simu, total_contacts_simu[i], label="{}".format(i))
                plt.legend()

            path_to_figure, tail = os.path.split(name)
            plt.savefig(self.input_path(path_to_figure,  "total_no_tangential_contacts_vs_time.pdf"))

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            # source =
            source, tail = os.path.split(self.input_path(name))
            # print("full source", source)
            # print(source[8:], "source 8 is")
            target_dir = "manuscript/figures/" + source[8:] + "/"
            try:
                os.makedirs(target_dir)
            except FileExistsError:
                pass

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


if __name__ == '__main__':
    PROBLEMS = [
        # Image generator
        Test01ElasticNormalImpactOfTwoIdenticalParticles,
        Test02ElasticNormalImpactParticleWall,
        Test03NormalParticleWallDifferentCOR,
        Test04ElasticNormalImpactParticleWall,

        # Some tests to test the DEM algorithm
        Tst03MultipleParticlesContacts
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
