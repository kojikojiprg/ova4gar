import os
from logging import Logger
from typing import Dict

import cv2
import numpy as np
import yaml
from group.group_analyzer import GroupAnalyzer
from individual.individual_analyzer import IndividualAnalyzer
from keypoint.visualization import draw_skeleton, put_frame_num
from tqdm import tqdm
from utility import json_handler
from utility.video import Capture, Writer, concat_field_with_frame


class Visalizer:
    def __init__(self, args, logger: Logger):
        # open config file
        with open(args.cfg_path) as f:
            cfg = yaml.safe_load(f)
        self._group_keys = cfg["group"]["indicator"].keys()
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

        self._logger.info(f"=> loading video from {video_path}.")
        self._logger.info(f"=> writing video into {out_path} while processing.")
        for frame_num in tqdm(range(video_capture.frame_count)):
            frame_num += 1  # frame_num = (1, ...)
            frame = video_capture.read()

            # write keypoint video
            if self._do_keypoint:
                kps_data_each_frame = [
                    kps for kps in kps_data if kps["frame"] == frame_num
                ]
                frame = put_frame_num(frame, frame_num)
                for kps in kps_data_each_frame:
                    frame = draw_skeleton(frame, kps["id"], np.array(kps["keypoints"]))
                kps_video_writer.write(frame)

            # write individual video
            if self._do_individual:
                ind_data_each_frame = [
                    kps for kps in ind_data if kps["frame"] == frame_num
                ]
                IndividualAnalyzer.write_video(
                    ind_video_writer, ind_data_each_frame, frame, self._field
                )

            # write group video
            if self._do_group:
                grp_data_each_frame = [
                    kps for kps in grp_data if kps["frame"] == frame_num
                ]
                GroupAnalyzer.write_video(
                    grp_writers, grp_data_each_frame, frame, self._field
                )

        # release memory
        del video_capture, kps_video_writer, ind_video_writer, grp_writers

    def _load_json(self, data_dir: str, name: str):
        json_path = os.path.join(data_dir, ".json", f"{name}.json")
        self._logger.info(f"=> loading {name} data from {json_path}")
        return json_handler.load(json_path)
