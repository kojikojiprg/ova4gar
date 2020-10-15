from common import keypoint as kp
from common import database
from human.functions import FUNC_DICT
import numpy as np


class Model:
    def __init__(self, start_frame_num):
        self.start_frame_num = start_frame_num
        self.keypoints_lst = kp.KeypointsList()
        self.data_dict = {k: [] for k in FUNC_DICT.keys()}

    def append(self, keypoints):
        self.keypoints_lst.append(kp.Keypoints(keypoints))
        for k in self.data_dict.keys():
            if keypoints is not None:
                self.data_dict[k].append(FUNC_DICT[k](keypoints))
            else:
                self.data_dict[k].append(np.nan)


def model(tracking_db, human_db):
    datas = tracking_db.select(database.TRACKING_TABLE.name)
    for row in datas:
