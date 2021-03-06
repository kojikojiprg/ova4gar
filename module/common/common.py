import os
import sys
import numpy as np

if 'module' in os.getcwd() or 'notebooks' in os.getcwd():
    root = '../'
else:
    root = './'

sys.path.append(root + 'module/')

data_dir = root + 'data/'
out_dir = root + 'out/'
db_dir = root + 'db/'

homo = {
    'record': [
        np.float32([[541, 300], [1141, 473], [1110, 1400], [-250, 800]]),
        np.float32([[0, 0], [438, 0], [438, 678], [0, 678]])
    ],
    '05': [
        np.float32([[378, 264], [910, 357], [926, 609], [91, 444]]),
        np.float32([[3, 90], [581, 90], [581, 400], [3, 400]])
    ],
    '09': [
        np.float32([[588, 118], [1191, 290], [756, 1057], [9, 395]]),
        np.float32([[3, 90], [494, 3], [581, 574], [3, 573]])
    ],
}
