import os


def make_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not exist.

    Parameters:
    directory_path (str): The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


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


def get_files_rigid_bodies(directory):
    # =====================================
    # start: get the files and sort
    # =====================================
    files = [filename for filename in os.listdir(directory) if filename.startswith("rigid_bodies") and filename.endswith("h5") ]
    files.sort()
    files_num = []
    for f in files:
        f_last = f[13:]
        files_num.append(int(f_last[:-3]))
    files_num.sort()

    sorted_files = []
    for num in files_num:
        sorted_files.append("rigid_bodies_" + str(num) + ".h5")
    files = sorted_files
    return files
