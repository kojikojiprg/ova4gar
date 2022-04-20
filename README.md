# Overview
研究用プロジェクト

# Installation
1. Install python requirements.
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu115
```

2, Update submodules.
```
git submodule update --init
```

3. Install [COCOAPI](https://github.com/cocodataset/cocoapi) into Python packages.  
Note: You can delete COCOAPI after you installed.
```
git clone https://github.com/cocodataset/cocoapi.git submodules/cocoapi
cd ./submodules/cocoapi/PythonAPI
make install
```

4. Install [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose.git) into Python packages.  
Note: You can also delete CrowdPose after you installed.
```
cd ../../../  # go back root of the project

git clone https://github.com/Jeff-sjtu/CrowdPose.git submodules/crowdpose
cd ./submodules/crowdpose/crowdpose-api/PythonAPI
sh install.sh
```

5. Download pretrained models from [Higher-HRNet model zoo](https://drive.google.com/drive/folders/1bdXVmYrSynPLSk5lptvgyQ8fhziobD50).
And store the model into ```./models/hrnet/```.
