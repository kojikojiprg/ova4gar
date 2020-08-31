import os
import sys

if 'python' in os.getcwd():
    root = '../'
else:
    root = './'

sys.path.append(root + 'python/common/')
sys.path.append(root + 'python/analysis/')
sys.path.append(root + 'python/tracking/')
data_dir = root + 'data/'
out_dir = root + 'out/'
db_dir = root + 'db/'


TRACKING_TABLE_NAME = 'Tracking'
TRACKING_TABLE_COLS = {
    'Person_ID': 'integer',
    'Frame_No': 'integer',
    'Keypoint': 'array'
}
