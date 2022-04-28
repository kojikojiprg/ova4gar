import sys
from logging import Logger
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torchvision
from numpy.typing import NDArray

from higher_hrnet.lib.config import cfg, check_config, update_config
from higher_hrnet.lib.core.group import HeatmapParser
from higher_hrnet.lib.core.inference import aggregate_results, get_multi_stage_outputs
from higher_hrnet.lib.models import pose_higher_hrnet
from higher_hrnet.lib.utils.transforms import (
    get_final_preds,
    get_multi_scale_size,
    resize_align_multi_scale,
)


class HigherHRNetDetecter:
    def __init__(self, cfg_path: str, logger: Logger, opts: list = []):
        # update config
        args = SimpleNamespace(**{"cfg": cfg_path, "opts": opts})
        self.cfg = cfg
        update_config(self.cfg, args)
        check_config(self.cfg)

        self.logger: Logger = logger

        # cudnn related setting
        torch.backends.cudnn.benchmark = self.cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = self.cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = self.cfg.CUDNN.ENABLED

        model = pose_higher_hrnet.get_pose_net(self.cfg, is_train=False)

        self.logger.info(
            "=> loading hrnet model from {}".format(self.cfg.TEST.MODEL_FILE)
        )
        model.load_state_dict(torch.load(self.cfg.TEST.MODEL_FILE), strict=True)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(model, device_ids=self.cfg.GPUS)
            self.model.cuda()
        else:
            self.model.cpu()

        self.model.eval()

        if self.cfg.MODEL.NAME == "pose_hourglass":
            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.parser = HeatmapParser(self.cfg)

    def __del__(self):
        # release memory
        del self.model, self.logger, self.cfg, self.transforms, self.parser

    def predict(self, image: NDArray):
        base_size, center, scale = get_multi_scale_size(
            image, self.cfg.DATASET.INPUT_SIZE, 1.0, min(self.cfg.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps: Any = None
            tags_list: list = []
            for s in sorted(self.cfg.TEST.SCALE_FACTOR, reverse=True):
                input_size = self.cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(self.cfg.TEST.SCALE_FACTOR)
                )
                image_resized = self.transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    self.cfg,
                    self.model,
                    image_resized,
                    self.cfg.TEST.FLIP_TEST,
                    self.cfg.TEST.PROJECT2IMAGE,
                    base_size,
                )

                final_heatmaps, tags_list = aggregate_results(
                    self.cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(self.cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = self.parser.parse(
                final_heatmaps, tags, self.cfg.TEST.ADJUST, self.cfg.TEST.REFINE
            )

        grouped = get_final_preds(
            grouped,
            center,
            scale,
            [final_heatmaps.size(3), final_heatmaps.size(2)],
        )
        kps = self._get_unique(grouped)
        return kps

    @staticmethod
    def _get_unique(grouped):
        kps = np.array(grouped)[:, :, :3]
        unique_kps = np.empty((0, 17, 3))

        for i in range(len(kps)):
            found_overlap = False

            for j in range(len(unique_kps)):
                found_overlap = True in (kps[i, :, :2] == unique_kps[j, :, :2])
                if found_overlap:
                    if np.mean(kps[i, :, 2]) > np.mean(unique_kps[j, :, 2]):
                        # select one has more confidence score
                        unique_kps[j] = kps[i]
                    break

            if not found_overlap:
                # if there aren't overlapped
                unique_kps = np.append(unique_kps, [kps[i]], axis=0)

        return unique_kps
