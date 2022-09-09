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
    def __init__(self, cfg_path: str, logger: Logger, device: str, opts: list = []):
        # update config
        args = SimpleNamespace(**{"cfg": cfg_path, "opts": opts})
        self.cfg = cfg
        update_config(self.cfg, args)
        check_config(self.cfg)

        self.logger: Logger = logger
        self.device = device

        # cudnn related setting
        torch.backends.cudnn.benchmark = self.cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = self.cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = self.cfg.CUDNN.ENABLED

        self.logger.info(
            "=> loading hrnet model from {}".format(self.cfg.TEST.MODEL_FILE)
        )
        self.model = pose_higher_hrnet.get_pose_net(self.cfg, is_train=False)
        self.model.load_state_dict(torch.load(self.cfg.TEST.MODEL_FILE), strict=True)
        self.model.to(device)

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
                image_resized = image_resized.unsqueeze(0).to(self.device)

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
        kps = np.array(grouped)[:, :, :3]
        return kps
