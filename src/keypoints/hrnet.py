from __future__ import absolute_import, division, print_function

import os
import pprint
import sys
from logging import Logger
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torchvision
from tqdm import tqdm

sys.path.append("./src/")
from utils.json_handler import dump
from utils.video import Capture, Writer

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
                "=> loading model from {}".format(self.cfg.TEST.MODEL_FILE)
            )
            model.load_state_dict(torch.load(self.cfg.TEST.MODEL_FILE), strict=True)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(model, device_ids=self.cfg.GPUS)
            self.model.cuda()
        else:
            self.model.cpu()

        self.model.eval()

        self.parser = HeatmapParser(self.cfg)

    def __del__(self):
        del self.model, self.logger, self.cfg, self.parser

    def predict(self, video_path: str, data_dir: str):
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

        # create video capture
        video_capture = Capture(video_path)
        assert (
            video_capture.is_opened
        ), f"{video_path} does not exist or is wrong file type."

        # create video writer
        out_path = os.path.join(data_dir, "video", "hrnet.mp4")
        video_writer = Writer(out_path, video_capture.fps, video_capture.size)
        self.logger.info(f"=> writing video into {out_path} while processing.")

        # prepair json data list
        json_path = os.path.join(data_dir, "json", "keypoints.json")
        json_data = []

        data_loader, test_dataset = make_test_dataloader(video_capture)
        pbar = tqdm(total=len(test_dataset))
        for frame_num, (rets, images) in enumerate(data_loader):
            if not rets[0]:
                self.logger.info(
                    f"=> couldn't read frame number {frame_num} on video {video_path}."
                )
                break

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
                grouped, scores = self.parser.parse(
                    final_heatmaps, tags, self.cfg.TEST.ADJUST, self.cfg.TEST.REFINE
                )

                final_results = get_final_preds(
                    grouped,
                    center,
                    scale,
                    [final_heatmaps.size(3), final_heatmaps.size(2)],
                )

            self._write_video(video_writer, image, final_results)  # write video

            # append result
            for idx, kps in enumerate(final_results):
                data = {
                    "frame": frame_num + 1,
                    "person": idx + 1,
                    "keypoints": np.array(kps)[:, :3],
                }
                json_data.append(data)

            pbar.update()

        pbar.close()

        self.logger.info(f" => Writing json file into {json_path}.")
        self._write_json(json_data, json_path)

        # release memory
        del (video_capture, video_writer, data_loader, test_dataset, json_data, data)

    def _write_video(self, writer: Writer, image, results):
        # add keypoints to image
        for person in results:
            color = np.random.randint(0, 255, size=3).tolist()
            image = add_joints(image, person, color)

        writer.write(image)

    def _write_json(self, json_data, json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        dump(json_data, json_path)
