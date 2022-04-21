import pprint
from distutils.log import Log
from logging import Logger
from types import SimpleNamespace
from typing import Dict, List

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
        # opts.img_size = img_size
        opts.resume = model_path

        self.logger = logger
        self.logger.info(f"=> unitrack config: {pprint.pformat(opts)}")

        self.logger.info(f"=> loading unitrack model from {opts.resume}")
        self.tracker = PoseAssociationTracker(opts)

        self.transforms = T.Compose(
            [T.ToTensor(), T.Normalize(opts.im_mean, opts.im_std)]
        )

    def __del__(self):
        del self.tracker, self.transforms

    def update(self, img: NDArray, img0: NDArray, kps: List[list]):
        img = img / 255.0
        img = self.transforms(img)

        obs = [self._cvt_kp2ob(kp) for kp in kps]

        tracks = self.tracker.update(img, img0, obs)
        for t in tracks:
            if not isinstance(t.pose[0], list):
                t.pose = self._cvt_ob2kp(t.pose)

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
