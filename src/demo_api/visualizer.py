import gc
import os
from logging import Logger
from typing import Dict

import cv2
import yaml
from tqdm import tqdm
from utility import json_handler
from utility.video import Capture, Writer, concat_field_with_frame
from visualize import individual as ind_vis
from visualize import keypoint as kps_vis
from visualize.group import GroupVisualizer
from visualize.util import delete_time_bar, get_size


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

        # group visualizer
        self._grp_vis = GroupVisualizer(self._group_keys)

        self._do_keypoint = not args.without_keypoint
        self._do_individual = not args.without_individual
        self._do_group = not args.without_group

        self._delete_height = args.delete_height

    def write_video(self, video_path: str, data_dir: str):
        # load data from json file
        kps_data = self._load_json(data_dir, "keypoints")
        ind_data = self._load_json(data_dir, "individual")
        grp_data = self._load_json(data_dir, "group")

        # create video capture
        self._logger.info(f"=> loading video from {video_path}.")
        video_capture = Capture(video_path)
        assert (
            video_capture.is_opened
        ), f"{video_path} does not exist or is wrong file type."

        # delete time bar
        tmp_frame = video_capture.read()[1]
        video_capture.set_pos_frame_count(0)
        tmp_frame = delete_time_bar(tmp_frame, self._delete_height)

        out_paths = []
        # create video writer for keypoints results
        if self._do_keypoint:
            out_path = os.path.join(data_dir, "video", "keypoints.mp4")
            kps_video_writer = Writer(
                out_path, video_capture.fps, tmp_frame.shape[1::-1]
            )
            out_paths.append(out_path)

        # create video writer for individual results
        size = get_size(tmp_frame, self._field)
        if self._do_individual:
            out_path = os.path.join(data_dir, "video", "individual.mp4")
            ind_video_writer = Writer(out_path, video_capture.fps, size)
            out_paths.append(out_path)

        # create video writer for group results
        if self._do_group:
            grp_writers: Dict[str, Writer] = {}
            for key in self._group_keys:
                out_path = os.path.join(data_dir, "video", f"{key}.mp4")
                grp_writers[key] = Writer(out_path, video_capture.fps, size)
                out_paths.append(out_path)

        if len(out_paths) == 0:
            self._logger.warning("=> There are no writing videos.")
            return

        self._logger.info(f"=> writing video into {out_paths}.")
        for frame_num in tqdm(range(video_capture.frame_count)):
            frame_num += 1  # frame_num = (1, ...)
            ret, frame = video_capture.read()
            frame = delete_time_bar(frame, self._delete_height)

            # write keypoint video
            frame = kps_vis.write_frame(frame, kps_data, frame_num, self._delete_height)
            if self._do_keypoint:
                kps_video_writer.write(frame)

            # write individual video
            field = ind_vis.write_field(ind_data, self._field.copy(), frame_num)
            if self._do_individual:
                frame_tmp = concat_field_with_frame(frame.copy(), field)
                ind_video_writer.write(frame_tmp)

            # write group video
            if self._do_group:
                for key in self._group_keys:
                    field_tmp = self._grp_vis.write_field(
                        key, frame_num, grp_data, field.copy()
                    )
                    frame_tmp = concat_field_with_frame(frame.copy(), field_tmp)
                    grp_writers[key].write(frame_tmp)

        # release memory
        del video_capture
        if self._do_keypoint:
            del kps_video_writer
        if self._do_individual:
            del ind_video_writer
        if self._do_group:
            del grp_writers
        gc.collect()

    def _load_json(self, data_dir: str, name: str):
        json_path = os.path.join(data_dir, ".json", f"{name}.json")
        self._logger.info(f"=> loading {name} data from {json_path}")
        return json_handler.load(json_path)
