import gc
import os
from glob import glob
from logging import Logger
from typing import Dict, List

import cv2
import yaml
from group.group import Group
from tqdm import tqdm
from utility.activity_loader import load_group
from utility.json_handler import load
from utility.video import Capture, Writer, concat_field_with_frame
from visualize.group import GroupVisualizer
from visualize.individual import write_field as ind_write_field
from visualize.keypoint import write_frame as kps_write_frame


class PassingAnalyzer:
    def __init__(self, cfg_path: str, logger: Logger):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        with open(cfg["config_path"]["group"], "r") as f:
            self._grp_cfg = yaml.safe_load(f)

        self._field = cv2.imread("image/field.png")

        self._logger = logger
        self._grp_vis = GroupVisualizer(["passing"])

    def _calc_start2end(
        self, group: Group, th_duration: int, th_interval: int
    ) -> Dict[str, list]:
        passing_dict = group.passing

        result_dict: Dict[str, list] = {}
        for pair_key, passing_frame_nums in tqdm(passing_dict.items()):
            result_dict[pair_key] = []

            pre_frame_num = start_frame_num = passing_frame_nums[0]
            for frame_num in passing_frame_nums[1:]:
                if frame_num - pre_frame_num > th_interval:
                    # difference between current and previous is over interval
                    if pre_frame_num - start_frame_num > th_duration:
                        # append result beyond with duration
                        result_dict[pair_key].append((start_frame_num, pre_frame_num))

                    start_frame_num = frame_num  # update start frame number

                pre_frame_num = frame_num  # update previous frame number
            else:
                # process for last frame number
                if pre_frame_num - start_frame_num > th_duration:
                    # append result beyond with duration
                    result_dict[pair_key].append((start_frame_num, pre_frame_num))

        return result_dict

    def extract_results(
        self, room_num: str, surgery_num: str, th_duration: int, th_interval: int
    ) -> List[Dict[str, list]]:
        data_dir = os.path.join("data", room_num, surgery_num)
        data_dirs = sorted(glob(os.path.join(data_dir, "*")))
        for i in range(len(data_dirs)):
            if data_dirs[i].endswith("passing") or data_dirs[i].endswith("attention"):
                del data_dirs[i]
        self._logger.info(f"=> data directories: {data_dirs}")

        results = []
        for data_dir in data_dirs:
            self._logger.info(f"=> load passing result from {data_dir}")
            json_path = os.path.join(data_dir, ".json", "group.json")
            if os.path.exists(json_path):
                group = load_group(
                    json_path,
                    self._grp_cfg,
                    self._field,
                    self._logger,
                    only_data_loading=True,
                )
                results.append(self._calc_start2end(group, th_duration, th_interval))

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
        results: List[Dict[str, list]],
        margin_frame_num: int,
    ):
        for i, result_dict in enumerate(results):
            i += 1
            # load json
            data_dir = os.path.join("data", room_num, surgery_num, f"{i:02d}")
            self._logger.info(f"=> load json files from {data_dir}")
            kps_data, ind_data, grp_data = self._load_jsons(data_dir)

            # delete previous files
            self._logger.info("=> delete files extracted previous process")
            for p in glob(os.path.join(data_dir, "video", "passing", "*.mp4")):
                if os.path.isfile(p):
                    os.remove(p)

            # create capture
            self._logger.info(f"=> load surgery {i:02d}.mp4")
            video_path = os.path.join("video", room_num, surgery_num, f"{i:02d}.mp4")
            cap = Capture(video_path)

            # calc output size
            cmb_img = concat_field_with_frame(cap.read()[1], self._field)
            size = cmb_img.shape[1::-1]

            pair_keys = list(result_dict.keys())
            for pair_key in pair_keys:
                pair_result = result_dict[pair_key]
                self._logger.info(f"=> write passing result {pair_key}: {pair_result}")
                for j, (start_num, end_num) in enumerate(pair_result):
                    j += 1

                    # create video writer
                    out_path = os.path.join(
                        data_dir, "video", "passing", f"{pair_key}_{j:2d}.mp4"
                    )
                    wrt = Writer(out_path, cap.fps, size)

                    start_num = max(1, start_num - margin_frame_num)
                    end_num = min(cap.frame_count, end_num + margin_frame_num)

                    # write video
                    cap.set_pos_frame_count(start_num - 1)
                    for frame_num in tqdm(range(start_num, end_num + 1)):
                        ret, frame = cap.read()

                        frame = kps_write_frame(frame, kps_data, frame_num)
                        field_tmp = ind_write_field(
                            ind_data, self._field.copy(), frame_num
                        )
                        field_tmp = self._grp_vis.write_field(
                            "passing", frame_num, grp_data, field_tmp
                        )
                        frame = concat_field_with_frame(frame.copy(), field_tmp)
                        wrt.write(frame)

                    del wrt

            del cap
            gc.collect()