import gc
import os
from glob import glob
from logging import Logger
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from group.group import Group
from scipy import signal
from tqdm import tqdm
from utility.activity_loader import load_group
from utility.json_handler import load
from utility.video import Capture, Writer, concat_field_with_frame
from visualize.group import GroupVisualizer
from visualize.individual import write_field as ind_write_field
from visualize.keypoint import write_frame as kps_write_frame
from visualize.util import delete_time_bar, get_size


class AttentionAnalyzer:
    def __init__(self, cfg_path: str, logger: Logger):
        with open(cfg_path, "r") as f:
            self._grp_cfg = yaml.safe_load(f)

        self._field = cv2.imread("image/field.png")

        self._logger = logger
        self._grp_vis = GroupVisualizer(["attention"])

    def _calc_peaks(
        self, group: Group, th_interval: int, th_max_val: float
    ) -> List[Tuple[int, int]]:
        attention_dict = group.attention
        heatmaps = list(attention_dict.values())
        max_val = np.max(np.max(heatmaps, axis=1), axis=1)
        peaks = signal.find_peaks(max_val, height=th_max_val)[0]
        if len(peaks) == 0:
            return []

        result_lst: List[Tuple[int, int]] = []
        pre_frame_num = start_frame_num = peaks[0] + 1
        for frame_num in peaks[1:]:
            frame_num += 1

            if frame_num - pre_frame_num > th_interval:
                # difference between current and previous is over interval
                result_lst.append((start_frame_num, pre_frame_num))
                start_frame_num = frame_num  # update start frame number

            pre_frame_num = frame_num  # update previous frame number
        else:
            # process for last frame number
            result_lst.append((start_frame_num, pre_frame_num))

        return result_lst

    def extract_results(
        self, room_num: str, surgery_num: str, th_interval: int, th_max_val: float = 3.5
    ) -> List[List[Tuple[int, int]]]:
        data_dir = os.path.join("data", room_num, surgery_num)
        data_dirs = sorted(glob(os.path.join(data_dir, "*")))
        for i in range(len(data_dirs)):
            if data_dirs[i].endswith("passing") or data_dirs[i].endswith("attention"):
                del data_dirs[i]
        self._logger.info(f"=> data directories: {data_dirs}")

        results = []
        for data_dir in data_dirs:
            self._logger.info(f"=> load attention result from {data_dir}")
            json_path = os.path.join(data_dir, ".json", "group.json")
            if os.path.exists(json_path):
                group = load_group(
                    json_path,
                    self._grp_cfg,
                    self._field,
                    self._logger,
                    only_data_loading=True,
                )
                results.append(self._calc_peaks(group, th_interval, th_max_val))

                del group
                gc.collect()

        return results

    @staticmethod
    def _load_jsons(data_dir):
        json_path = os.path.join(data_dir, ".json", "keypoints.json")
        kps_data = load(json_path)
        json_path = os.path.join(data_dir, ".json", "individual.json")
        ind_data = load(json_path)
        json_path = os.path.join(data_dir, ".json", "group.json")
        grp_data = load(json_path)
        return kps_data, ind_data, grp_data

    def crop_videos(
        self,
        room_num: str,
        surgery_num: str,
        results: List[List[Tuple[int, int]]],
        margin_frame_num: int,
    ):
        for i, result_lst in enumerate(results):
            i += 1
            data_dir = os.path.join("data", room_num, surgery_num, f"{i:02d}")

            if len(result_lst) == 0:
                self._logger.info(f"=> skip writing result {data_dir}")
                continue

            # load json
            self._logger.info(f"=> load json files from {data_dir}")
            kps_data, ind_data, grp_data = self._load_jsons(data_dir)

            # delete previous files
            self._logger.info("=> delete files extracted previous process")
            for p in glob(os.path.join(data_dir, "video", "attention", "*.mp4")):
                if os.path.isfile(p):
                    os.remove(p)

            # create capture
            self._logger.info(f"=> load surgery {i:02d}.mp4")
            video_path = os.path.join("video", room_num, surgery_num, f"{i:02d}.mp4")
            cap = Capture(video_path)

            # calc output size
            tmp_frame = cap.read()[1]
            tmp_frame = delete_time_bar(tmp_frame)
            size = get_size(tmp_frame, self._field)

            self._logger.info(
                f"=> write attention result: {result_lst}"
            )
            for j, (start_num, end_num) in enumerate(result_lst):
                j += 1

                # create video writer
                out_path = os.path.join(
                    data_dir, "video", "attention", f"{i:02d}_{j:02d}.mp4"
                )
                wrt = Writer(out_path, cap.fps, size)

                start_num = max(1, start_num - margin_frame_num)
                end_num = min(cap.frame_count, end_num + margin_frame_num)

                # write video
                cap.set_pos_frame_count(start_num - 1)
                for frame_num in tqdm(range(start_num, end_num + 1)):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = delete_time_bar(frame)
                    frame = kps_write_frame(frame, kps_data, frame_num)
                    field_tmp = ind_write_field(
                        ind_data, self._field.copy(), frame_num
                    )
                    field_tmp = self._grp_vis.write_field(
                        "attention", frame_num, grp_data, field_tmp
                    )
                    frame = concat_field_with_frame(frame.copy(), field_tmp)
                    wrt.write(frame)

                del wrt

            del kps_data, ind_data, grp_data, cap
            gc.collect()
