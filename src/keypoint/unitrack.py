import pprint
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from torchvision.transforms import transforms as T

from data.video import letterbox
from tracker.mot.pose import PoseAssociationTracker


class UniTrackTracker:
    def __init__(self, cfg_path, logger, img_size=[640, 480], conf_thres=0.65):
        # set config
        opts = SimpleNamespace(**{})
        with open(cfg_path) as f:
            common_args = yaml.safe_load(f)
        for k, v in common_args["common"].items():
            setattr(opts, k, v)
        for k, v in common_args["posetrack"].items():
            setattr(opts, k, v)
        opts.img_size = img_size
        opts.conf_thres = conf_thres
        opts.resume = "models/unitrack/crw.pth"

        self.logger = logger
        self.logger.info(f"=> unitrack config: {pprint.pformat(opts)}")

        self.logger.info(f"=> loading unitrack model from {opts.resume}")
        self.tracker = PoseAssociationTracker(opts)

        self.transforms = T.Compose(
            [T.ToTensor(), T.Normalize(opts.im_mean, opts.im_std)]
        )

        self.w = img_size[0]
        self.h = img_size[1]

    def __del__(self):
        del self.tracker, self.transforms

    def update(self, img, kps):
        tmp_img = cv2.resize(img, (self.w, self.h))

        # Padded resize
        tmp_img, _, _, _ = letterbox(tmp_img, height=self.height, width=self.width)

        # Normalize RGB
        tmp_img = img[:, :, ::-1]
        tmp_img = np.ascontiguousarray(tmp_img, dtype=np.float32)

        tmp_img = tmp_img / 255.0
        tmp_img = self.transforms(tmp_img)

        obs = [self._cvt_kp(kp) for kp in kps]

        tracks = self.tracker.update(tmp_img, img, obs)
        for t in tracks:
            t.pose = self._cvt_ob2kp(t.pose)

        return tracks

    @staticmethod
    def _cvt_kp2ob(kp):
        # https://github.com/leonid-pishchulin/poseval
        return [
            {"id": [i], "x": [p[0]], "y": [p[1]], "score": [p[2]]}
            for i, p in enumerate(kp)
        ]

    @staticmethod
    def _cvt_ob2kp(ob):
        return [[p["x"][0], p["y"][0], p["score"][0]] for p in ob]
