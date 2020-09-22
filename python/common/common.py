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

