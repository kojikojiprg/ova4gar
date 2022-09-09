# Operating Room Surveillance Video Analysis for Group Activity Recognition

# Environment
- OS: Ubuntu-20.04
- Pyhon: 3.9.12
- CUDA: 11.5

# Installation
1. Install python requirements and submodules.
```
sh install.sh
```

2. Download pretrained models from [Higher-HRNet model zoo](https://drive.google.com/drive/folders/1bdXVmYrSynPLSk5lptvgyQ8fhziobD50).  
And store the model into ```./models/higher_hrnet/```.

3. Download pretrained models from [HRNet model zoo](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA).  
And store the model into ```./models/hrnet/```.

4. Download pretrained models from [UniTrack model zoo](https://github.com/Zhongdao/UniTrack/blob/main/docs/MODELZOO.md), BarlowTwins recommended.  
And store the model into ```./models/unitrack/```.


# Video Storage
You have to store srugery videos as follows.
- ROOM_NUM  
The operating room number
- DATE  
Date of surgery
- VIDEO*.mp4  
Video data whose name are not specified, but in order to time series.
And codec has to be mp4.

```
video
├── [ROOM_NUM]
│   └── [DATE]
│       ├── [VIDEO1].mp4
│       ├── [VIDEO2].mp4
│       ├── ...
```

# Demonstration
## Full surgery inference
```
python demo_surgery.py [-h] -rn ROOM_NUM -d DATE [-c CFG_PATH] [-wk] [-wi] [-wg]
```
### Optional Arguments:
- -rn ROOM_NUM, --room_num ROOM_NUM : 
operating room number
- -d DATE, --date DATE : 
date of surgery
- -h, --help : show this help message and exit
- -c CFG_PATH, --cfg_path CFG_PATH : 
Config file path.
- -wk, --without_keypoint : 
without keypoint extraction.
- -wi, --without_individual : 
without idividual analyzation.
- -wg, --without_group : 
without group analyzation.

## One Video Inference
```
python demo_file.py [-h] video_path data_dir -rn ROOM_NUM [-c CFG_PATH] [-wk] [-wi] [-wg]
```
### Positional Arguments:
- video_path : video file path 
- data_dir : data directory path where results are saved

### Optional Arguments:
- -h, --help : show this help message and exit
- -rn ROOM_NUM, --room_num ROOM_NUM : 
operating room number
- -c CFG_PATH, --cfg_path CFG_PATH : 
config file path.
- -wk, --without_keypoint : 
without keypoint extraction.
- -wi, --without_individual : 
without idividual analyzation.
- -wg, --without_group : 
without group analyzation.
