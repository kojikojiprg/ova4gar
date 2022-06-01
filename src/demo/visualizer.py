import gc
import os
from logging import Logger
from typing import Dict

import cv2
import yaml
from group.group import Group
from group.group_analyzer import GroupAnalyzer
from individual.individual_analyzer import IndividualAnalyzer
from tqdm import tqdm
from utility import json_handler
from utility.video import Capture, Writer, concat_field_with_frame

from visualize import keypoint


class Visalizer:
    def __init__(self, args, logger: Logger):
        # open config file
        with open(args.cfg_path) as f:
            cfg = yaml.safe_load(f)
        with open(cfg["config_path"]["group"]) as f:
            self._grp_cfg = yaml.safe_load(f)
        self._group_keys = self._grp_cfg.keys()
        self._logger = logger

        # homography
        self._field = cv2.imread(cfg["homography"]["field_path"])

        self._do_keypoint = not args.without_keypoint
        self._do_individual = not args.without_individual
        self._do_group = not args.without_group

    def visualize(self, video_path: str, data_dir: str):
        # create video capture
        video_capture = Capture(video_path)
        assert (
            video_capture.is_opened
        ), f"{video_path} does not exist or is wrong file type."

        # create video writer for keypoints results
        out_path = os.path.join(data_dir, "video", "keypoints.mp4")
        kps_video_writer = Writer(out_path, video_capture.fps, video_capture.size)

        # create video writer for individual results
        cmb_img = concat_field_with_frame(video_capture.read()[1], self._field)
        video_capture.set_pos_frame_count(0)
        size = cmb_img.shape[1::-1]
        out_path = os.path.join(data_dir, "video", "individual.mp4")
        ind_video_writer = Writer(out_path, video_capture.fps, size)

        # create video writer for group results
        grp_writers: Dict[str, Writer] = {}
        out_paths = []
        for key in self._group_keys:
            out_path = os.path.join(data_dir, "video", f"{key}.mp4")
            out_paths.append(out_path)
            grp_writers[key] = Writer(out_path, video_capture.fps, size)

        # load data from json file
        kps_data = self._load_json(data_dir, "keypoints")
        ind_data = self._load_json(data_dir, "individual")
        grp_data = self._load_json(data_dir, "group")
        group = Group(self._grp_cfg, self._field, self._logger, "cpu")
        group.from_json(grp_data)

        self._logger.info(f"=> loading video from {video_path}.")
        self._logger.info(f"=> writing video into {out_path} while processing.")
        for frame_num in tqdm(range(video_capture.frame_count)):
            frame_num += 1  # frame_num = (1, ...)
            ret, frame = video_capture.read()

            # write keypoint video
            frame = keypoint.write_frame(frame, kps_data, frame_num)
            if self._do_keypoint:
                kps_video_writer.write(frame)

            # write individual video
            ind_data_each_frame = [kps for kps in ind_data if kps["frame"] == frame_num]
            if self._do_individual:
                IndividualAnalyzer.write_video(
                    ind_video_writer, ind_data_each_frame, frame, self._field
                )

            # write group video
            if self._do_group:
                GroupAnalyzer.write_video(
                    grp_writers,
                    frame_num,
                    frame,
                    self._field,
                    ind_data_each_frame,
                    group,
                )

        # release memory
        del video_capture, kps_video_writer, ind_video_writer, grp_writers
        gc.collect()

    def _load_json(self, data_dir: str, name: str):
        json_path = os.path.join(data_dir, ".json", f"{name}.json")
        self._logger.info(f"=> loading {name} data from {json_path}")
        return json_handler.load(json_path)
