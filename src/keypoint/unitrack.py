import pprint
from logging import Logger
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import yaml
from numpy.typing import NDArray
from torchvision.transforms import transforms as T
from tracker.mot.pose import PoseAssociationTracker  # from unitrack


class UniTrackTracker:
    def __init__(
        self, cfg_path: str, logger: Logger, model_path: str = "models/unitrack/crw.pth"
    ):
        # set config
        opts = SimpleNamespace(**{})
        with open(cfg_path) as f:
            common_args = yaml.safe_load(f)
        for k, v in common_args["common"].items():
            setattr(opts, k, v)
        for k, v in common_args["posetrack"].items():
            setattr(opts, k, v)
        opts.resume = model_path

        self.logger = logger

        self.logger.info(f"=> loading unitrack model from {opts.resume}")
        self.tracker = PoseAssociationTracker(opts)

        self.transforms = T.Compose(
            [T.ToTensor(), T.Normalize(opts.im_mean, opts.im_std)]
        )

    def __del__(self):
        del self.tracker, self.transforms

    def update(self, img: NDArray, kps: List[list]):
        # Normalize RGB
        process_img = img[:, :, ::-1]
        process_img = np.ascontiguousarray(process_img, dtype=np.float32)

        process_img = process_img / 255.0
        process_img = self.transforms(process_img)

        obs = [self._cvt_kp2ob(kp) for kp in kps]

        tracks = self.tracker.update(process_img, img, obs)
        for t in tracks:
            if not isinstance(t.pose[0], list):
                t.pose = self._cvt_ob2kp(t.pose)
            else:
                tracks.remove(t)  # remove not updated track

        return tracks

    @staticmethod
    def _cvt_kp2ob(kp: List[list]):
        # https://github.com/leonid-pishchulin/poseval
        return [
            {"id": [i], "x": [p[0]], "y": [p[1]], "score": [p[2]]}
            for i, p in enumerate(kp)
        ]

    @staticmethod
    def _cvt_ob2kp(ob: List[Dict[str, list]]):
        return [[p["x"][0], p["y"][0], p["score"][0]] for p in ob]
