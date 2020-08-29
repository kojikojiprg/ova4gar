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
