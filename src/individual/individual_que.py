from __future__ import annotations

from typing import Any, Dict, Tuple, overload

import numpy as np
from keypoint.keypoint import Keypoints
from utility.functions import mahalanobis


class Que:
    def __init__(self, default):
        self._default = default
        self._size = default.size
        self._th_std = default.th_std
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
            item = np.average(que, axis=0)
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
            item = np.average(que[np.abs(distances - mean) < std * th_std], axis=0)

        return item


class KeypointQue(Que):
    def __init__(self, default):
        super().__init__(default)
        self._window = default.window
        self._th_conf = default.th_conf
        assert (
            self._size > self._window
        ), f"que_size:{self._size} > window:{self._window} is expected."

    def put_pop_kps(
        self,
        frame_num,
        pre_frame_num,
        new_kps,
        kps_dict: Dict[int, Keypoints],
    ) -> Tuple[Dict[int, Keypoints], Keypoints]:
        # if confidnce score < THRESHOLD then [np.nan, np.nan, np.nan]
        new_kps = np.array(new_kps)
        mask = np.where(new_kps.T[2] < self._th_conf)
        nan_array = np.full(new_kps.shape, np.nan)
        new_kps[mask] = nan_array[mask]

        # append keypoints dict
        kps_dict[frame_num] = Keypoints(new_kps)

        if frame_num - pre_frame_num >= self._size:
            self._que = []

        # fill copied keypoints for blank
        pre = np.array(kps_dict[pre_frame_num])
        for i in range(pre_frame_num, frame_num + 1):
            if i in kps_dict:
                # append current keypoints
                kps = np.array(kps_dict[i])
                pre = kps.copy()
            else:
                # copy and append pre keypoints
                kps_dict[i] = pre.copy()

        # fill nan each keypoint
        for i in range(0, len(self._size) - self._window + 1):
            # calc means of all keypoints
            copies = [
                kps_dict[j]
                for j in range(pre_frame_num + i, pre_frame_num + i + self._window)
            ]
            kps_means = np.nanmean(copies, axis=0)

            self._que.append(kps_means)
            if len(self._que) >= self._size:
                self._que = self._que[-self._size :]

        # update keypoints data
        for i, kps in enumerate(self._que):
            kps_dict[pre_frame_num + i] = kps

        # calc mean of que
        if len(self._que) < self._size:
            new_kps = np.nanmean(self._que, axis=0)
        else:
            new_kps = []
            tmp_que = np.transpose(self._que, (1, 0, 2))
            for i in range(len(new_kps)):
                new_kps.append(self._moving_average(tmp_que[i], self._th_std))
            new_kps = Keypoints(new_kps)

        return new_kps, kps_dict
