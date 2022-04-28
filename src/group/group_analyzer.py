import os
from logging import Logger
from typing import Any, Dict, List, Tuple

import numpy as np
from individual.individual import Individual
from individual.individual_analyzer import IndividualAnalyzer
from individual.visualization import visualize as individual_vis
from tqdm import tqdm
from utility import json_handler
from utility.video import Capture, Writer, concat_field_with_frame

from group.group import Group
from group.visualization import GroupVisualizer


class GroupAnalyzer:
    def __init__(self, cfg: dict, logger: Logger):
        self._cfg = cfg["group"]
        self._keys = list(self._cfg["indicator"].keys())
        self._logger = logger

        self._ind_defs = IndividualAnalyzer.load_default(cfg["individual"])
        self._visualizer = GroupVisualizer(self._keys)

    def analyze(self, data_dir: str, field: np.typing.NDArray):
        # load individual data from json file
        ind_json_path = os.path.join(data_dir, ".json", "individual.json")
        self._logger.info(f"=> load individual data from {ind_json_path}")
        inds, last_frame_num = self._load_individuals(ind_json_path)

        # create group class
        self._logger.info(f"=> construct group activity model for {data_dir}")
        group = Group(self._cfg, field, self._logger)

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
        writers: Dict[str, Writer] = {}
        for key in self._keys:
            out_path = os.path.join(data_dir, "video", f"{key}.mp4")
            video_writer = Writer(out_path, video_capture.fps, size)
            writers[key] = video_writer

        self._logger.info(f"=> writing video into {out_path} while processing")
        for frame_num in tqdm(range(last_frame_num)):
            inds_per_frame = [
                ind for ind in inds.values() if ind.exists_on_frame(frame_num)
            ]

            group.calc_indicator(frame_num, inds_per_frame)

            inds_video_data = [ind.to_dict(frame_num) for ind in inds_per_frame]
            _, frame = video_capture.read()
            self._write_video(writers, frame_num, frame, field, inds_video_data, group)

        # write json
        group_data = group.to_dict()
        grp_json_path = os.path.join(data_dir, ".json", "group.json")
        self._logger.info(f"=> write group data to {grp_json_path}")
        json_handler.dump(group_data, grp_json_path)

        del inds, group, group_data, video_capture, writers  # release memory

    def _load_individuals(
        self, ind_json_path: str
    ) -> Tuple[Dict[int, Individual], int]:
        inds = {}
        last_frame_num = 0
        json_data = json_handler.load(ind_json_path)
        for item in json_data:
            frame_num = item["frame"]
            if item["id"] not in inds:
                inds[item["id"]] = Individual(item["id"], self._ind_defs)
            inds[item["id"]].from_json(item, frame_num)
            last_frame_num = frame_num if frame_num > last_frame_num else last_frame_num

        return inds, last_frame_num

    def _write_video(
        self,
        writers: Dict[str, Writer],
        frame_num: int,
        frame: np.typing.NDArray,
        field: np.typing.NDArray,
        inds_data: List[Dict[str, Any]],
        group: Group,
    ):
        for key in self._keys:
            field_tmp = individual_vis(inds_data, field.copy())
            field_tmp = self._visualizer.visualize(
                key, frame_num, group.to_dict(), field_tmp
            )
            frame_tmp = concat_field_with_frame(frame.copy(), field_tmp)
            writers[key].write(frame_tmp)
