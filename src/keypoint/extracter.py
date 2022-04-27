import os
from logging import Logger
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from tracker.mot.basetrack import STrack  # from unitrack
from utility.json_handler import dump
from utility.video import Capture, Writer
from utility.visualization import draw_skeleton

from .dataset import make_test_dataloader
from .hrnet import HRNetDetecter
from .unitrack import UniTrackTracker


class Extractor:
    def __init__(self, cfg: dict, logger: Logger):
        self.logger = logger
        self.detector = HRNetDetecter(cfg["keypoint"]["hrnet_cfg_path"], logger)
        self.tracker = UniTrackTracker(cfg["keypoint"]["unitrack_cfg_path"], logger)

    def __del__(self):
        del self.detector, self.tracker, self.logger

    def predict(self, video_path: str, data_dir: str):
        # create video capture
        video_capture = Capture(video_path)
        assert (
            video_capture.is_opened
        ), f"{video_path} does not exist or is wrong file type."

        # create video writer
        out_path = os.path.join(data_dir, "video", "tracking.mp4")
        video_writer = Writer(out_path, video_capture.fps, video_capture.size)

        json_data = []
        self.logger.info(f"=> loading video from {video_path}.")
        data_loader = make_test_dataloader(video_capture)
        self.logger.info(f"=> writing video into {out_path} while processing.")
        for frame_num, imgs in enumerate(tqdm(data_loader)):
            assert 1 == imgs.size(0), "Test batch size should be 1"
            img = imgs[0].cpu().numpy()

            # do keypoints detection and tracking
            kps = self.detector.predict(img)
            tracks = self.tracker.update(img, kps)

            self._write_video(video_writer, img, tracks)  # write video

            # append result
            for t in tracks:
                data = {
                    "frame": frame_num + 1,
                    "id": t.track_id,
                    "keypoints": np.array(t.pose),
                }
                json_data.append(data)
                del data  # release memory

            del imgs, img, kps, tracks  # release memory

        json_path = os.path.join(data_dir, "json", "keypoints.json")
        self.logger.info(f"=> writing json file into {json_path}.")
        dump(json_data, json_path)

        # release memory
        del (video_capture, video_writer, data_loader, json_data)

    def _write_video(self, writer: Writer, image: NDArray, tracks: List[STrack]):
        # add keypoints to image
        for t in tracks:
            image = draw_skeleton(image, t.track_id, t.pose)

        writer.write(image)
