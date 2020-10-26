import os
import sys
import numpy as np

if 'python' in os.getcwd():
    root = '../'
else:
    root = './'

sys.path.append(root + 'python/')

data_dir = root + 'data/'
out_dir = root + 'out/'
db_dir = root + 'db/'

limit = {
    'basketball': [[0, 340], [np.inf, 660]],
    'tenis': [[0, 0], [np.inf, np.inf]],
}

homo = {
    'basketball': [
        np.float32([[210, 364], [1082, 362], [836, 488], [438, 489]]),
        np.float32([[24, 24], [568, 24], [383, 232], [205, 232]])
    ],
    'tenis': [
        np.float32([[381, 201], [897, 201], [1104, 567], [171, 567]]),
        np.float32([[27, 24], [160, 24], [160, 238], [27, 238]])
    ],
}
