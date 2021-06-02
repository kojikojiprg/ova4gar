import os
import sys
import numpy as np

if os.getcwd().endswith('src'):
    root = '../'
elif 'notebooks' in os.getcwd():
    if os.getcwd().endswith('notebooks'):
        root = '../'
    else:
        root = '../../'
else:
    root = './'

sys.path.append(os.path.join(root, 'src/'))

data_dir = os.path.join(root, 'data/')

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
        np.float32([[738, 277], [801, 300], [567, 462], [497, 430]]),
        np.float32([[336, 320], [400, 320], [400, 514], [336, 514]])
    ],
}
