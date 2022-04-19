# Overview
研究用プロジェクト

# Installation
1. Install python requirements.
```
poetry install
```

2, Update submodules
```
git submodule update --init
```

3. Install [COCOAPI](https://github.com/cocodataset/cocoapi)
```
cd ./submodules/cocoapi/PythonAPI
make install
```

4. Install [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose.git)
```
cd ../../../  # go back root of the project
cd ./submodules/crowdpose/crowdpose-api/PythonAPI
sh install.sh
```
