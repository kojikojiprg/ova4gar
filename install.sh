#!/bin/sh

# install crowdpose
git clone https://github.com/Jeff-sjtu/CrowdPose.git
cd CrowdPose/crowdpose-api/PythonAPI
sh install.sh
cd ../../../  # go back root of the project
rm -rf CrowdPose

# install unitrack
cd ./submodules/unitrack
python setup.py
sed -i 'utils/jaccard_similarity_score/d' mask.py  # sklearn >= 0.23 changed this function name
cd ../../  # go back root of the project

# install hrnet
cd ./submodules/hrnet/lib
make
cd ../../../  # go back root of the project
