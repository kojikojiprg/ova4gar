import sys
from ast import arg
from logging import Logger

import cv2
import torch
import yaml

sys.path.append("src")
from group.group_analyzer import GroupAnalyzer
from individual.individual_analyzer import IndividualAnalyzer
from keypoint.extracter import Extractor
from utility.device import get_device
from utility.transform import Homography


class InferenceModel:
    def __init__(self, args, logger: Logger):
        # open config file
        with open(args.cfg_path) as f:
            cfg = yaml.safe_load(f)

        # homography
        self._homo, self._field = self._create_homography(cfg, args.room_num)

        self._do_keypoint = not args.without_keypoint
        self._do_individual = not args.without_individual
        self._do_group = not args.without_group

        device = get_device(args)
        if self._do_keypoint:
            self.extractor = Extractor(cfg, logger, device)
        if self._do_individual:
            self.individual_analyzer = IndividualAnalyzer(cfg, logger)
        if self._do_group:
            self.group_analyzer = GroupAnalyzer(cfg, logger, device)

    def __del__(self):
        if hasattr(self, "extractor"):
            del self.extractor
        if hasattr(self, "individual_analyzer"):
            del self.individual_analyzer
        if hasattr(self, "group_analyzer"):
            del self.group_analyzer

    def _create_homography(self, cfg, room_num):
        homo_cfg = cfg["homography"]
        field = cv2.imread(homo_cfg["field_path"])

        p_video = homo_cfg[room_num]["video"]
        p_field = homo_cfg[room_num]["field"]
        homo = Homography(p_video, p_field, field.shape)
        return homo, field

    def inference(self, video_path, data_dir):
        if self._do_keypoint:
            self.extractor.predict(video_path, data_dir)

        if self._do_individual:
            self.individual_analyzer.analyze(data_dir, self._homo, self._field)

        if self._do_group:
            self.group_analyzer.analyze(data_dir, self._field)
