import os
import sys
import numpy as np

cur_dir = os.getcwd()
if 'notebooks' in os.getcwd() or 'src' in os.getcwd():
    root = cur_dir
    dirs = cur_dir.split('/')

    if 'notebooks' in os.getcwd():
        dir_num = len(dirs) - dirs.index('notebooks')
    elif 'src' in os.getcwd():
        dir_num = len(dirs) - dirs.index('src')

    for _ in range(dir_num):
        root = os.path.dirname(root)
else:
    root = cur_dir

sys.path.append(os.path.join(root, 'src/'))

data_dir = os.path.join(root, 'data/')
model_dir = os.path.join(root, 'model/')

homo = {
    '02': [
        np.float32([[426, 553], [494, 510], [773, 638], [718, 692]]),
        np.float32([[336, 320], [400, 320], [400, 514], [336, 514]])
    ],
    '05': [
        np.float32([[378, 264], [910, 357], [926, 609], [91, 444]]),
        np.float32([[3, 90], [581, 90], [581, 400], [3, 400]])
    ],
    '08': [
        np.float32([[508, 502], [829, 502], [829, 704], [508, 704]]),
        np.float32([[336, 320], [400, 320], [400, 514], [336, 514]])
    ],
    '09': [
        np.float32([[738, 277], [801, 300], [567, 462], [497, 430]]),
        np.float32([[336, 320], [400, 320], [400, 514], [336, 514]])
    ],
    '09_0304': [
        np.float32([[738, 277], [801, 300], [567, 462], [497, 430]]),
        np.float32([[356, 342], [422, 342], [422, 548], [356, 548]])
    ],
}
