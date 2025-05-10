"""
"""
import shutil
import sys
import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import h5py

from utils import make_directory_if_not_exists

# =============================================
# 1. compile the code the code
# =============================================
os.system('cd build' + '&& make -j 12 ')
os.system('cd ../')

# =====================================================
# 2. copy the executable to the output directory and run
# =====================================================
# create the output directory if it doesn't exists
output_dir = sys.argv[1]
make_directory_if_not_exists(output_dir)

shutil.copy('./build/examples/Tst01BDEMTwoParticlesTensile3D', output_dir)

# executable args
cli_args = ' '.join(element.split("=")[1] for element in sys.argv[2:])
os.system('cd ' + output_dir + '&& ./Tst01BDEMTwoParticlesTensile3D ' + cli_args)

# # =====================================================
# # 3. post process the output file and plot
# # =====================================================
# cabana = h5py.File(output_dir + "/particles_0.h5", "r")

# cabana_x = cabana['positions'][:, 0]
# cabana_y = cabana['positions'][:, 1]
# # cabana_sin_appr = cabana['pressure'][:]
# # cabana_sin_analytical = cabana['wij'][:]

# # res_npz = os.path.join(output_dir, "results.npz")
# # np.savez(res_npz,
# #          x=cabana_x,
# #          sin_appr=cabana_sin_appr,
# #          sin_analytical=cabana_sin_analytical)

# # # only plot some of the points
# # step = int(len(cabana_x) / 20)
# # cabana_x_plot = cabana_x[::step]
# # cabana_sin_appr_plot = cabana_sin_appr[::step]
# # cabana_sin_analytical_plot = cabana_sin_analytical[::step]

# # plt.plot(cabana_x_plot, cabana_sin_analytical_plot, "^-", label="Analytical")
# fig, ax = plt.subplots()
# ax.scatter(cabana_x, cabana_y, label="SPH approximation")
# # ax.plot(x, y)

# # Set Equal Aspect Ratio
# ax.set_aspect('equal')


# # plt.legend()
# # res_plot = os.path.join(output_dir, "sin_appr.pdf")
# # plt.savefig(res_plot)
# # plt.axes().set_aspect('equal', 'box')
# plt.show()

# from utils import get_files
# files = get_files(output_dir)
# print(files)

# # TODO add one more condition to skip this plot and go to direct comparision plots
# if len(files) > 0:
#     # print(directory_name+files[0])
#     fn_simu = []
#     time_simu = []
#     for f in files:
#         f = h5py.File(input_path(name, f), "r")
#         fn_simu.append(f["forces"][1][0] / 1e3)
#         time_simu.append(f.attrs["Time"] / 1e-6)

#     # save the simulated data in a npz file in the case folder
#     res_npz = os.path.join(self.input_path(name, "results.npz"))
#     np.savez(res_npz,
#                 time_simu=time_simu,
#                 fn_simu=fn_simu)

#     plt.scatter(time_analy, fn_analy, label="Analytical")
#     plt.plot(time_simu, fn_simu, "^-", label="Cabana DEM solver")
#     plt.legend()
#     plt.savefig(self.input_path(name, "fn_vs_time.pdf"))
