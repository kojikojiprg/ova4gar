from __future__ import annotations

from typing import Any, Dict, Tuple, overload

import numpy as np
from keypoint.keypoint import Keypoints
from utility.functions import mahalanobis


class Que:
    def __init__(self, default):
        self._default = default
        self._size = default["que_size"]
        self._th_std = default["th_std"]
        self._que: list = []

    def put_pop(self, new_item: Any) -> Any:
        self._que.append(new_item)
        if len(self._que) < self._size:
            ret_item = np.average(self._que, axis=0)
        else:
            self._que = self._que[-self._size :]
            ret_item = self._moving_average(self._que, self._th_std)

        return ret_item

    @staticmethod
    def _moving_average(que, th_std):
        que = np.array(que)

        if np.any(np.std(que, axis=0) < 1.0):
            # 分布の中身がほぼ同じのとき
            return np.average(que, axis=0)
        else:
            if que.ndim < 2:
                # 各点の中心からの距離を求める
                mean = np.nanmean(que)
                distances = np.abs(que - mean)
            else:
                # 各点の中心からのマハラノビス距離を求める
                distances = [mahalanobis(x, que) for x in que]

            # 中心からの距離の平均と分散を求める
            mean = np.nanmean(distances)
            std = np.std(distances)

            # 外れ値を除去した平均値を値とする
            exclude_outlier = que[np.abs(distances - mean) < std * th_std]
            if len(exclude_outlier) > 0:
                return np.average(exclude_outlier, axis=0)
            else:
                return None


class KeypointQue(Que):
    def __init__(self, default):
        super().__init__(default)
        self._window = default["window"]
        self._th_conf = default["th_conf"]
        assert (
            self._size > self._window
        ), f"que_size:{self._size} > window:{self._window} is expected."

    def put_pop_kps(
        self,
        frame_num: int,
        pre_frame_num: int,
        new_kps: Keypoints,
        kps_dict: Dict[int, Keypoints],
    ) -> Tuple[Dict[int, Keypoints], Keypoints]:
        # if confidnce score < th_conf then [np.nan, np.nan, np.nan]
        new_kps = np.array(new_kps)
        mask = np.where(new_kps.T[2] < self._th_conf)
        nan_array = np.full(new_kps.shape, np.nan)
        new_kps[mask] = nan_array[mask]

        # append keypoints dict
        kps_dict[frame_num] = Keypoints(new_kps)
        if pre_frame_num == 0:
            # when initial frame
            return Keypoints(new_kps), kps_dict

        # fill copied keypoints for blank
        pre = np.array(kps_dict[pre_frame_num])
        for i in range(pre_frame_num + 1, frame_num):
            if i in kps_dict:
                # append current keypoints
                kps = np.array(kps_dict[i])
                pre = kps.copy()
            else:
                # copy and append pre keypoints for filling blank
                kps_dict[i] = pre.copy()

        if len(kps_dict) <= self._window:
            kps_lst = list(kps_dict.values())
            kps_means = np.nanmean(kps_lst, axis=0)
            self._que.append(kps_means)
        else:
            # fill nan each keypoint
            for i in range(pre_frame_num, frame_num + 1 - self._window):
                # calc means of all keypoints
                kps_window = [
                    kps_dict[i + j] for j in range(self._window)
                ]
                kps_means = np.nanmean(kps_window, axis=0)

                self._que.append(kps_means)
                if len(self._que) >= self._size:
                    self._que = self._que[-self._size :]

        # calc mean of que
        if len(self._que) < self._size:
            ret_kps = np.nanmean(self._que, axis=0)
        else:
            ret_kps = []
            tmp_que = np.transpose(self._que, (1, 0, 2))
            for i in range(len(new_kps)):
                ret_pt = self._moving_average(tmp_que[i], self._th_std)
                if ret_pt is not None:
                    ret_kps.append(ret_pt)
                else:
                    ret_kps.append(np.full((3,), np.nan))

        ret_kps = Keypoints(ret_kps)
        return ret_kps, kps_dict
