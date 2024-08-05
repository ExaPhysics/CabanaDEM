import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


# =====================================
# start: get the files and sort
# =====================================

directory_name = "../outputs/test01_elastic_normal_impact_of_two_identical_particles/case_1/"
files = [filename for filename in os.listdir(directory_name) if filename.startswith("particles") and filename.endswith("xmf") ]
files.sort()
files_num = []
for f in files:
    f_last = f[10:]
    files_num.append(int(f_last[:-4]))
files_num.sort()

sorted_files = []
for num in files_num:
    sorted_files.append("particles_" + str(num) + ".xmf")
# print(sorted_files)
files = sorted_files
# =====================================
# end: get the files and sort
# =====================================

# # print(directory_name+files[0])
# for f in files:
#     f = h5py.File(directory_name+f[:-3]+'h5', "r")
#     # print(np.array(f["radius"]))
#     x = np.array(f["forces"][0][0])
#     # y = np.array(f["forces"][0][0])
#     print(x)
#     # print(y)
# # plt.scatter(x, y, label="SPH appr")
# # plt.legend()
# # plt.savefig("colliding_fluid_blocks.png")
# # plt.show()
# # plt.plot()

# print(directory_name+files[0])
f = h5py.File(directory_name+files[10][:-3]+'h5', "r")
# print(np.array(f["radius"]))
x = np.array(f["forces"][0][0])
# x = np.array(f["forces"][0][0])
# y = np.array(f["forces"][0][0])
print(x)
# print(y)
# plt.scatter(x, y, label="SPH appr")
# plt.legend()
# plt.savefig("colliding_fluid_blocks.png")
# plt.show()
# plt.plot()

# import h5py

# f = h5py.File(directory_name+files[10][:-3]+'h5', "r"file_name, mode)
