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

limit = {
    'record': [[0, 0], [np.inf, np.inf]],
    '05': [[0, 0], [np.inf, np.inf]],
    '09': [[0, 0], [np.inf, np.inf]],
}

homo = {
    'record': [
        np.float32([[541, 300], [1141, 473], [1110, 1400], [-250, 800]]),
        np.float32([[0, 0], [438, 0], [438, 678], [0, 678]])
    ],
    '05': [
        np.float32([[378, 264], [910, 357], [926, 609], [91, 444]]),
        np.float32([[3, 94], [581, 94], [581, 400], [3, 400]])
    ],
    '09': [
        np.float32([[1317, 219], [2199, 538], [2349, 669], [1090, 226]]),
        np.float32([[90, 3], [494, 3], [581, 91], [3, 91]])
    ],
}
