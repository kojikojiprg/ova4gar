# Overview
研究用プロジェクト

# Installation
1. Install python requirements.
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu115
pip install cython_bbox lap # cython_bbox requires Cython, lap requires numpy
```

2, Update submodules.
```
git submodule update --init
```

3. Install [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose.git) into Python packages.  
Note: You can also delete CrowdPose after you installed.
```
git clone https://github.com/Jeff-sjtu/CrowdPose.git submodules/crowdpose
cd ./submodules/crowdpose/crowdpose-api/PythonAPI
sh install.sh
cd ../../../../  # go back root of the project
```

4. Install [UniTrack](https://github.com/Zhongdao/UniTrack) for this project.
```
cd ./submodules/unitrack
python setup.py
cd ../../  # go back root of the project
```

5. Install [HRNet]().  
```
cd ./submodules/hrnet/lib
make
cd ../../../
```

6. Download pretrained models from [Higher-HRNet model zoo](https://drive.google.com/drive/folders/1bdXVmYrSynPLSk5lptvgyQ8fhziobD50).  
And store the model into ```./models/higher_hrnet/```.

7. Download pretrained models from [HRNet model zoo](https://drive.google.com/drive/folders/14p2l1u19bLOm5p6esKyolYJyfE1_inLv).  
And store the model into ```./models/hrnet/```.

8. Download pretrained models from [UniTrack model zoo](https://github.com/Zhongdao/UniTrack/blob/main/docs/MODELZOO.md), BarlowTwins recommended.  
And store the model into ```./models/unitrack/```.

9. Delete specific Python code in UniTrack.
```
cd ./submodules/unitrack/utils
sed -i '/jaccard_similarity_score/d' mask.py  # sklearn >= 0.23 changed this function name
cd ../../../  # go back root of the project
```
