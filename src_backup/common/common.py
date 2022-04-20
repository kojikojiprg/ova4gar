import os
import sys

pf_is_win = os.name == "nt"
delimiter = "\\" if pf_is_win else "/"


def split_path(path):
    return path.replace("\\", "/").split("/")


cur_dir = os.getcwd()
if "notebooks" in os.getcwd() or "src" in os.getcwd():
    root = cur_dir
    dirs = split_path(cur_dir)

    if "notebooks" in os.getcwd():
        dir_num = len(dirs) - dirs.index("notebooks")
    elif "src" in os.getcwd():
        dir_num = len(dirs) - dirs.index("src")

    for _ in range(dir_num):
        root = os.path.dirname(root)
else:
    root = cur_dir

sys.path.append(os.path.join(root, "src"))

data_dir = os.path.join(root, "data")
model_dir = os.path.join(root, "model")
