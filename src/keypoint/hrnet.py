import pprint
from logging import Logger
from types import SimpleNamespace
from typing import Any

import models  # from hrnet
import torch
import torchvision
from config import cfg, check_config, update_config  # from hrnet
from core.group import HeatmapParser  # from hrnet
from core.inference import aggregate_results, get_multi_stage_outputs  # from hrnet
from fp16_utils.fp16util import network_to_half  # from hrnet
from numpy.typing import NDArray
from utils.transforms import get_final_preds  # from hrnet
from utils.transforms import get_multi_scale_size, resize_align_multi_scale


class HRNetDetecter:
    def __init__(self, cfg_path: str, opts: list, logger: Logger):
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

        model = eval("models." + self.cfg.MODEL.NAME + ".get_pose_net")(
            self.cfg, is_train=False
        )

        if self.cfg.FP16.ENABLED:
            model = network_to_half(model)

        if self.cfg.TEST.MODEL_FILE:
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
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, self.cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )

        with torch.no_grad():
            final_heatmaps: Any = None
            tags_list: list = []
            input_size = self.cfg.DATASET.INPUT_SIZE

            image_resized, center, scale = resize_align_multi_scale(
                image, input_size, 1, 1
            )
            image_resized = self.transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            _, heatmaps, tags = get_multi_stage_outputs(
                self.cfg,
                self.model,
                image_resized,
                self.cfg.TEST.FLIP_TEST,
                self.cfg.TEST.PROJECT2IMAGE,
                base_size,
            )

            final_heatmaps, tags_list = aggregate_results(
                self.cfg, 1, final_heatmaps, tags_list, heatmaps, tags
            )

            final_heatmaps = final_heatmaps / float(len(self.cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = self.parser.parse(
                final_heatmaps, tags, self.cfg.TEST.ADJUST, self.cfg.TEST.REFINE
            )

            pred = get_final_preds(
                grouped,
                center,
                scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)],
            )

        return pred
