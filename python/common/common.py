import os
import sys

if 'python' in os.getcwd():
    root = '../'
else:
    root = './'

sys.path.append(root + 'python/')

data_dir = root + 'data/'
out_dir = root + 'out/'
db_dir = root + 'db/'


TRACKING_TABLE_NAME = 'Tracking'
TRACKING_TABLE_COLS = {
    'Person_ID': 'integer',
    'Frame_No': 'integer',
    'Keypoints': 'array',
    'Vector': 'array',
}

VECTOR_TABLE_NAME = 'Vector'
VECTOR_TABLE_COLS = {
    'Person_ID': 'integer',
    'Frame_No': 'integer',
    'Vector': 'array',
}

MOVE_HAND_TABLE_NAME = 'Move_Hand'
MOVE_HAND_TABLE_COLS = {
    'Person_ID': 'integer',
    'Frame_No': 'integer',
    'Move_Hand': 'float',
}

DENSITY_TABLE_NAME = 'Density'
DENSITY_TABLE_COLS = {
    'Frame_No': 'integer',
    'Density': 'array',
}
