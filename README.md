# Overview
研究用プロジェクト

# Installation
1. Install python requirements.
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu115

# cython_bbox requires Cython, lap requires numpy
pip install cython_bbox lap 
```

2, Update submodules.
```
git submodule update --init
```

3. Install submodules
```
sh install.sh
```

4. Download pretrained models from [Higher-HRNet model zoo](https://drive.google.com/drive/folders/1bdXVmYrSynPLSk5lptvgyQ8fhziobD50).  
And store the model into ```./models/higher_hrnet/```.

5. Download pretrained models from [HRNet model zoo](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA).  
And store the model into ```./models/hrnet/```.

6. Download pretrained models from [UniTrack model zoo](https://github.com/Zhongdao/UniTrack/blob/main/docs/MODELZOO.md), BarlowTwins recommended.  
And store the model into ```./models/unitrack/```.
