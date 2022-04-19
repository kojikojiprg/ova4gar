from __future__ import absolute_import, division, print_function

import pprint
import sys
from logging import Logger
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from .dataset import make_test_dataloader

sys.path.append("./submodules/hrnet/lib/")
import models
from config import cfg, check_config, update_config
from core.group import HeatmapParser
from core.inference import aggregate_results, get_multi_stage_outputs
from fp16_utils.fp16util import network_to_half
from utils.transforms import (
    get_final_preds,
    get_multi_scale_size,
    resize_align_multi_scale,
)
from utils.utils import get_model_summary
from utils.vis import add_joints


class HRNetExtractor:
    def __init__(self, cfg_path: str, opts: list, logger: Logger):
        # update config
        args = SimpleNamespace(**{"cfg": cfg_path, "opts": opts})
        self.cfg = cfg
        update_config(self.cfg, args)
        check_config(self.cfg)

        self.logger: Logger = logger
        self.logger.info(pprint.pformat(args))
        self.logger.info(self.cfg)

        # cudnn related setting
        torch.backends.cudnn.benchmark = self.cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = self.cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = self.cfg.CUDNN.ENABLED

        model = eval("models." + self.cfg.MODEL.NAME + ".get_pose_net")(
            self.cfg, is_train=False
        )

        dump_input = torch.rand(
            (1, 3, self.cfg.DATASET.INPUT_SIZE, self.cfg.DATASET.INPUT_SIZE)
        )
        self.logger.info(get_model_summary(model, dump_input, verbose=self.cfg.VERBOSE))

        if self.cfg.FP16.ENABLED:
            model = network_to_half(model)

        if self.cfg.TEST.MODEL_FILE:
            self.logger.info(
                "=> loading model from {}".format(self.cfg.TEST.MODEL_FILE)
            )
            model.load_state_dict(torch.load(self.cfg.TEST.MODEL_FILE), strict=True)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(model, device_ids=self.cfg.GPUS)
            self.model.cuda()
        else:
            self.model.cpu()

        self.model.eval()

    def predict(self, video_path: str, data_dir: str):
        data_loader, test_dataset = make_test_dataloader(video_path)

        if self.cfg.MODEL.NAME == "pose_hourglass":
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        parser = HeatmapParser(self.cfg)
        all_preds = []
        all_scores = []

        pbar = tqdm(total=len(test_dataset))
        for images in data_loader:
            assert 1 == images.size(0), "Test batch size should be 1"

            image = images[0].cpu().numpy()
            # size at scale 1.0
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
                    image_resized = transforms(image_resized)
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
                grouped, scores = parser.parse(
                    final_heatmaps, tags, self.cfg.TEST.ADJUST, self.cfg.TEST.REFINE
                )

                final_results = get_final_preds(
                    grouped,
                    center,
                    scale,
                    [final_heatmaps.size(3), final_heatmaps.size(2)],
                )

            if self.cfg.TEST.LOG_PROGRESS:
                pbar.update()

            # add keypoints to image
            for person in final_results:
                color = np.random.randint(0, 255, size=3).tolist()
                add_joints(image, person, color)
            # save_debug_images(self.cfg, image_resized, None, None, outputs, prefix)

            all_preds.append(final_results)
            all_scores.append(scores)

        if self.cfg.TEST.LOG_PROGRESS:
            pbar.close()

        name_values, _ = test_dataset.evaluate(
            self.cfg, all_preds, all_scores, data_dir
        )

        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, self.cfg.MODEL.NAME)
        else:
            self._print_name_value(name_values, self.cfg.MODEL.NAME)

    def _print_name_value(self, name_value, full_arch_name):
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        self.logger.info(
            "| Arch " + " ".join(["| {}".format(name) for name in names]) + " |"
        )
        self.logger.info("|---" * (num_values + 1) + "|")

        if len(full_arch_name) > 15:
            full_arch_name = full_arch_name[:8] + "..."
        self.logger.info(
            "| "
            + full_arch_name
            + " "
            + " ".join(["| {:.3f}".format(value) for value in values])
            + " |"
        )
