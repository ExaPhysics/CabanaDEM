#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

from itertools import cycle, product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores
# from pysph.solver.utils import load, get_files
from automan.api import (Automator, Simulation, filter_cases, filter_by_name)

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


def get_files_at_given_times(files, times):
    from pysph.solver.utils import load
    result = []
    count = 0
    for f in files:
        data = load(f)
        t = data['solver_data']['t']
        if count >= len(times):
            break
        if abs(t - times[count]) < t * 1e-8:
            result.append(f)
            count += 1
        elif t > times[count]:
            result.append(f)
            count += 1
    return result


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result


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
    #     self.move_figures()

    # def move_figures(self):
    #     import shutil
    #     import os

    #     for name in self.case_info:
    #         source = self.input_path(name)

    #         target_dir = "manuscript/figures/" + source[8:] + "/"
    #         os.makedirs(target_dir)
    #         # print(target_dir)

    #         file_names = os.listdir(source)

    #         for file_name in file_names:
    #             # print(file_name)
    #             if file_name.endswith((".jpg", ".pdf", ".png")):
    #                 # print(target_dir)
    #                 shutil.copy(os.path.join(source, file_name), target_dir)


if __name__ == '__main__':
    PROBLEMS = [
        # Image generator
        Test01ElasticNormalImpactOfTwoIdenticalParticles
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
