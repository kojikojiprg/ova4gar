import gc
import sys
from logging import Logger

import cv2
import torch
import yaml

sys.path.append("src")
from group.group_analyzer import GroupAnalyzer
from individual.individual_analyzer import IndividualAnalyzer
from utility.transform import Homography

from demo_api.visualizer import Visalizer


class InferenceModel:
    def __init__(self, args, logger: Logger):
        # open config file
        with open(args.cfg_path) as f:
            cfg = yaml.safe_load(f)
        with open(cfg["config_path"]["individual"]) as f:
            ind_cfg = yaml.safe_load(f)
        with open(cfg["config_path"]["group"]) as f:
            grp_cfg = yaml.safe_load(f)

        # homography
        self._homo, self._field = self._create_homography(cfg, args.room_num)

        self._do_individual = args.individual
        self._do_group = args.group
        self._write_video = args.video

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._do_individual:
            self.individual_analyzer = IndividualAnalyzer(ind_cfg, logger)
        if self._do_group:
            self.group_analyzer = GroupAnalyzer(grp_cfg, ind_cfg, logger, device)
        if self._write_video:
            self.visualizer = Visalizer(args, logger)

    def __del__(self):
        if hasattr(self, "extractor"):
            del self.extractor
        if hasattr(self, "individual_analyzer"):
            del self.individual_analyzer
        if hasattr(self, "group_analyzer"):
            del self.group_analyzer
        if hasattr(self, "visualizer"):
            del self.visualizer
        gc.collect()

    def _create_homography(self, cfg, room_num):
        homo_cfg = cfg["homography"]
        field = cv2.imread(homo_cfg["field_path"])

        p_video = homo_cfg[room_num]["video"]
        p_field = homo_cfg[room_num]["field"]
        homo = Homography(p_video, p_field, field.shape)
        return homo, field

    def inference(self, video_path, data_dir):
        if self._do_individual:
            self.individual_analyzer.analyze(data_dir, self._homo)

        if self._do_group:
            self.group_analyzer.analyze(data_dir, self._field)

        if self._write_video:
            self.visualizer.write_video(video_path, data_dir)
