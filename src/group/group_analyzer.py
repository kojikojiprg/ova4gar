import gc
import os
from logging import Logger
from typing import Any, Dict, List

import torch
from individual.visualization import visualize as individual_vis
from numpy.typing import NDArray
from tqdm import tqdm
from utility import json_handler
from utility.activity_loader import load_individuals
from utility.video import Capture, Writer, concat_field_with_frame

from group.group import Group
from group.visualization import GroupVisualizer


class GroupAnalyzer:
    def __init__(self, grp_cfg: dict, ind_cfg: dict, logger: Logger, device: str):
        self._ind_cfg = ind_cfg
        self._grp_cfg = grp_cfg
        self._keys = list(self._grp_cfg.keys())
        self._logger = logger
        self._device = device

        self._visualizer = GroupVisualizer(self._keys)

    def __del__(self):
        torch.cuda.empty_cache()
        del self._visualizer
        gc.collect()

    def analyze(self, data_dir: str, field: NDArray, writing_video: bool = False):
        # load individual data from json file
        ind_json_path = os.path.join(data_dir, ".json", "individual.json")
        self._logger.info(f"=> load individual data from {ind_json_path}")
        inds, last_frame_num = load_individuals(ind_json_path, self._ind_cfg)

        # create group class
        self._logger.info(f"=> construct group activity model for {data_dir}")
        group = Group(self._grp_cfg, field, self._logger, self._device)

        if writing_video:
            # create video capture
            video_path = os.path.join(data_dir, "video", "keypoints.mp4")
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
            out_paths = []
            for key in self._keys:
                out_path = os.path.join(data_dir, "video", f"{key}.mp4")
                out_paths.append(out_path)
                video_writer = Writer(out_path, video_capture.fps, size)
                writers[key] = video_writer

            self._logger.info(f"=> writing video into {out_paths} while processing")

        for frame_num in tqdm(range(last_frame_num)):
            inds_per_frame = [
                ind for ind in inds.values() if ind.exists_on_frame(frame_num)
            ]

            group.calc_indicator(frame_num, inds_per_frame)

            if writing_video:
                inds_video_data = [ind.to_dict(frame_num) for ind in inds_per_frame]
                _, frame = video_capture.read()
                self.write_video(
                    writers, frame_num, frame, field, inds_video_data, group
                )

        # write json
        group_data = group.to_dict()
        grp_json_path = os.path.join(data_dir, ".json", "group.json")
        self._logger.info(f"=> write group data to {grp_json_path}")
        json_handler.dump(group_data, grp_json_path)

        # release memory
        torch.cuda.empty_cache()
        del inds, group, group_data
        if writing_video:
            del video_capture, writers
        gc.collect()

    def write_video(
        self,
        writers: Dict[str, Writer],
        frame_num: int,
        frame: NDArray,
        field: NDArray,
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
