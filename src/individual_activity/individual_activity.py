import numpy as np
from common.functions import standardize
from common.json import IA_FORMAT, START_IDX
from common.keypoint import Keypoints, body

from individual_activity.indicator import INDICATOR_DICT, calc_keypoints


class IndividualActivity:
    def __init__(self, activity_id, homo):
        self.id = activity_id
        self.tracking_points = {}
        self.keypoints = {}
        self.keypoints_que = {b: [] for b in body.keys()}
        self.indicator_dict = {k: {} for k in INDICATOR_DICT.keys()}
        self.que_dict = {k: [] for k in INDICATOR_DICT.keys()}
        self.homo = homo

    def calc_indicator(self, frame_num, keypoints):
        if keypoints is None:
            return
        keypoints = Keypoints(keypoints)

        self.tracking_points[frame_num] = keypoints.get_middle("Hip")
        self.keypoints[frame_num], self.keypoints_que = calc_keypoints(
            keypoints, self.keypoints_que
        )

        for k in self.indicator_dict.keys():
            indicator, self.que_dict[k] = INDICATOR_DICT[k](
                keypoints, self.homo, self.que_dict[k]
            )

            self.indicator_dict[k][frame_num] = indicator

    def get_indicator(self, key, frame_num):
        if key not in IA_FORMAT:
            raise KeyError

        if frame_num in self.indicator_dict[key]:
            return self.indicator_dict[key][frame_num]
        else:
            return None

    def get_indicators(self, key):
        if key not in IA_FORMAT:
            raise KeyError

        return self.indicator_dict[key]

    def get_keypoints(self, key, frame_num):
        if key not in body:
            raise KeyError

        if frame_num in self.keypoints:
            return self.keypoints[frame_num][body[key]]
        else:
            return None

    def get_keypoints_dict(self, window=3, is_std=False) -> dict:
        # fill nan
        min_frame_num = min(self.keypoints.keys())
        max_frame_num = max(self.keypoints.keys())

        copy_kps_lst = []
        pre = np.array(self.keypoints[min_frame_num])
        for frame_num in range(min_frame_num + 1, max_frame_num + 1):
            if frame_num in self.keypoints:
                kps = np.array(self.keypoints[frame_num])
                if True in np.isnan(kps):
                    # 一部のnanを前フレームからコピー
                    kps = np.where(np.isnan(kps), pre, kps).copy()
                copy_kps_lst.append(kps)
                pre = kps.copy()
            else:
                # 前のフレームからコピー
                copy_kps_lst.append(pre)

        # 残ったnanは移動平均で穴埋め
        ma_kps_lst = []
        for i in range(0, len(copy_kps_lst) - window):
            means = np.nanmean(copy_kps_lst[i : i + window], axis=0)
            for kps in copy_kps_lst[i : i + window]:
                if True in np.isnan(kps):
                    kps = np.where(np.isnan(kps), means, kps).copy()

                if len(ma_kps_lst) <= i + window:
                    ma_kps_lst.append(kps)

        ma_kps_dict = {}
        for i, kps in enumerate(ma_kps_lst):
            ma_kps_dict[min_frame_num + i] = np.array(kps)

        if is_std:
            # standardize
            std_kps_dict = {}
            for frame_num, kps in ma_kps_dict.items():
                std_kps_dict[frame_num] = np.array(standardize(kps))

            return std_kps_dict
        else:
            return ma_kps_dict

    def to_json(self, frame_num):
        data = {}
        data[IA_FORMAT[0]] = self.id
        data[IA_FORMAT[1]] = frame_num
        data[IA_FORMAT[2]] = self.tracking_points[frame_num]
        data[IA_FORMAT[3]] = self.keypoints[frame_num]

        for k in IA_FORMAT[START_IDX:]:
            indicator = self.indicator_dict[k][frame_num]
            if indicator is not None:
                data[k] = indicator.tolist()
            else:
                data[k] = None

        return data
