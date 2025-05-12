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

# =====================================================
# 3. post process the output file and plot
# =====================================================
cabana = h5py.File(output_dir + "/particles_0.h5", "r")

total_no_bonds = cabana['total_no_bonds'][:]
print(total_no_bonds)
# cabana_sin_appr = cabana['pressure'][:]
# cabana_sin_analytical = cabana['wij'][:]

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

# from utils import get_files, get_files_rigid_bodies
# files = get_files(output_dir)
# print(files)

# # TODO add one more condition to skip this plot and go to direct comparision plots
# if len(files) > 0:
#     # print(directory_name+files[0])
#     x_0_simu = []
#     x_1_simu = []
#     time_simu = []
#     for f_name in files:
#         f_path = os.path.join(output_dir, f_name)
#         f = h5py.File(f_path, "r")
#         # print(f["rot_mat_cm"][:])
#         x_0_simu.append(f["x"][0][0])
#         x_1_simu.append(f["x"][1][0])
#         time_simu.append(f.attrs["Time"])
# else:
#     sys.exit("Files are empty")

# # # save the simulated data in a npz file in the case folder
# # res_npz = os.path.join(os.path.join(output_dir, "results.npz"))
# # np.savez(res_npz,
# #          time_simu=time_simu,
# #          R_0_simu=R_0_simu)

# plt.plot(time_simu, x_0_simu, label="Rotation matrix index 0")
# plt.plot(time_simu, x_1_simu, label="Rotation matrix index 1")
# plt.show()
# plt.legend()
# plt.savefig(os.path.join(output_dir, "time_vs_rot_mat_00.pdf"))
