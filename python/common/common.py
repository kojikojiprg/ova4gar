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


class table:
    def __init__(self, name, cols):
        self.name = name
        self.cols = cols


TRACKING_TABLE = table(
    'Tracking',
    {
        'Person_ID': 'integer',
        'Frame_No': 'integer',
        'Keypoints': 'array',
        'Vector': 'array',
        'Average': 'array',
    }
)

VECTOR_TABLE = table(
    'Vector',
    {
        'Person_ID': 'integer',
        'Frame_No': 'integer',
        'Average': 'array',
        'Vector': 'array',
    }
)

MOVE_HAND_TABLE = table(
    'Move_Hand',
    {
        'Person_ID': 'integer',
        'Frame_No': 'integer',
        'Point': 'array',
        'Move_Hand': 'float',
    }
)

DENSITY_TABLE = table(
    'Density',
    {
        'Frame_No': 'integer',
        'Density': 'array',
    }
)

TABLE_DICT = {
    TRACKING_TABLE.name: TRACKING_TABLE,
    VECTOR_TABLE.name: VECTOR_TABLE,
    MOVE_HAND_TABLE.name: MOVE_HAND_TABLE,
    DENSITY_TABLE.name: DENSITY_TABLE,
}
