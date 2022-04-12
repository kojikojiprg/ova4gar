# Overview
研究用プロジェクト

# Installation
1. Install python requirements.
```
poetry install
```

2, Update submodules
```
git submodule update
```

3. Install [COCOAPI](https://github.com/cocodataset/cocoapi)
```
cd submodule/cocoapi/PythonAPI

# Install into global site-packages
make install

# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python3 setup.py install --user
```
