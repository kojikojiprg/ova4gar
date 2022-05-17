from ast import arg
import sys
from logging import Logger

import cv2
import torch
import yaml

sys.path.append("src")
from group.group_analyzer import GroupAnalyzer
from individual.individual_analyzer import IndividualAnalyzer
from keypoint.extracter import Extractor
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

        if torch.cuda.is_available():
            device = "cuda"
            if args.gpu is not None:
                device += f":{args.gpu}"
        else:
            device = "cpu"

        if self._do_keypoint:
            self.extractor = Extractor(cfg, logger, device)
        if self._do_individual:
            self.individual_anlyzer = IndividualAnalyzer(cfg, logger)
        if self._do_group:
            self.group_anlyzer = GroupAnalyzer(cfg, logger, device)

    def __del__(self):
        if isinstance(self.extractor, Extractor):
            del self.extractor
        if isinstance(self.individual_anlyzer, IndividualAnalyzer):
            del self.individual_anlyzer
        if isinstance(self.group_anlyzer, GroupAnalyzer):
            del self.group_anlyzer

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
            self.individual_anlyzer.analyze(data_dir, self._homo, self._field)

        if self._do_group:
            self.group_anlyzer.analyze(data_dir, self._field)
