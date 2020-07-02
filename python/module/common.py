import os
import sys

if 'python' in os.getcwd():
    root = '../'
else:
    root = './'

sys.path.append(root + 'python/module/')

data_dir = root + 'data/'
out_dir = root + 'out/'
