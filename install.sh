#!/bin/sh

# install python libraries
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu115

# cython_bbox requires Cython, lap requires numpy.
pip install cython_bbox lap

# init submodules
git submodule update --init

# install crowdpose
git clone https://github.com/Jeff-sjtu/CrowdPose.git
cd CrowdPose/crowdpose-api/PythonAPI
sh install.sh
cd ../../../  # go back root of the project
rm -rf CrowdPose

# install unitrack
cd ./submodules/unitrack
python setup.py
# sklearn >= 0.23 changed this function name
sed -i '/jaccard_similarity_score/d' utils/mask.py
cd ../../  # go back root of the project

# install hrnet
cd ./submodules/hrnet/lib
make
cd ../../../  # go back root of the project
