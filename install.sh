#!/bin/sh

# install python libraries
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# cython_bbox requires Cython, lap requires numpy.
pip install cython_bbox lap
