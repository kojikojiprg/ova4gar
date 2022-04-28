import os
from logging import Logger
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm
from utility import json_handler
from utility.transform import Homography
from utility.video import Capture, Writer, concat_field_with_frame

from individual.individual import Individual
from individual.visualization import visualize


class IndividualAnalyzer:
    def __init__(self, cfg: dict, logger: Logger):
        # load config
        cfg = cfg["individual"]
        self._defaults: Dict[str, Dict[str, Any]] = self.load_default(cfg)

        self._logger = logger

    @staticmethod
    def load_default(cfg: dict):
        defaults: Dict[str, Dict[str, Any]] = {"indicator": {}, "keypoint": {}}
        for indicator_key, item in cfg["indicator"].items():
            defaults["indicator"][indicator_key] = {}
            for key, val in item["default"].items():
                defaults["indicator"][indicator_key][key] = val
        for key, val in cfg["keypoint"]["default"].items():
            defaults["keypoint"][key] = val
        return defaults

    def analyze(self, data_dir: str, homo: Homography, field: np.typing.NDArray):
        # create video capture
        video_path = os.path.join(data_dir, "video", "tracking.mp4")
        self._logger.info(f"=> loading video from {video_path}.")
        video_capture = Capture(video_path)
        assert (
            video_capture.is_opened
        ), f"{video_path} does not exist or is wrong file type."

        # create video writer
        cmb_img = concat_field_with_frame(video_capture.read()[1], field)
        video_capture.set_pos_frame_count(0)
        size = cmb_img.shape[1::-1]
        out_path = os.path.join(data_dir, "video", "individual.mp4")
        video_writer = Writer(out_path, video_capture.fps, size)

        # load keypoints data from json file
        kps_json_path = os.path.join(data_dir, "json", "keypoints.json")
        self._logger.info(f"=> loading keypoint data from {kps_json_path}")
        keypoints_data = json_handler.load(kps_json_path)

        individuals: Dict[int, Individual] = {}
        json_data: List[Dict[str, Any]] = []
        pre_frame_num = 1
        self._logger.info(f"=> writing video into {out_path} while processing")
        for data in tqdm(keypoints_data):
            frame_num = data["frame"]
            pid = data["id"]
            keypoints = data["keypoints"]

            # obtain individual
            if pid not in individuals:
                individuals[pid] = Individual(pid, self._defaults)
            ind = individuals[pid]

            # calc indicators of individual
            ind.calc_indicator(frame_num, keypoints, homo)

            # create and append json data
            output = ind.to_dict(frame_num)
            json_data.append(output)

            # when frame next, write video frame
            if pre_frame_num < frame_num:
                video_data = [
                    ind.to_dict(pre_frame_num)
                    for ind in individuals.values()
                    if ind.exists_on_frame(pre_frame_num)
                ]
                _, frame = video_capture.read()
                self._write_video(video_writer, video_data, frame, field)
                del video_data  # release memory
                pre_frame_num = frame_num  # update pre_frame_num
        else:
            video_data = [
                ind.to_dict(frame_num)
                for ind in individuals.values()
                if ind.exists_on_frame(frame_num)
            ]
            _, frame = video_capture.read()
            self._write_video(video_writer, video_data, frame, field)
            del video_data  # release memory

        # write json
        ind_json_path = os.path.join(data_dir, "json", "individual.json")
        self._logger.info(f"=> writing individual data to {ind_json_path}")
        json_handler.dump(json_data, ind_json_path)

        # release memory
        del video_capture, video_writer, keypoints_data, individuals, json_data

    @staticmethod
    def _write_video(
        writer: Writer,
        data: List[Dict[str, Any]],
        frame: np.typing.NDArray,
        field: np.typing.NDArray,
    ):
        field_tmp = visualize(data, field.copy())
        frame = concat_field_with_frame(frame, field_tmp)
        writer.write(frame)
