import os
from logging import Logger
from typing import List

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from unitrack.tracker.mot.basetrack import STrack
from utility.json_handler import dump
from utility.video import Capture, Writer

from keypoint.api.higher_hrnet import HigherHRNetDetecter
from keypoint.api.hrnet import HRNetDetecter
from keypoint.api.unitrack import UniTrackTracker
from keypoint.dataset import make_test_dataloader
from keypoint.visualization import draw_skeleton, put_frame_num


class Extractor:
    def __init__(self, cfg: dict, logger: Logger):
        self._logger = logger

        cfg = cfg["keypoint"]
        detector_name = cfg["detector"]
        if detector_name == "higher_hrnet":
            self._detector = HigherHRNetDetecter(
                cfg["cfg_path"]["higher_hrnet"], logger
            )
        elif detector_name == "hrnet":
            self._detector = HRNetDetecter(cfg["cfg_path"]["hrnet"], logger)
        else:
            raise KeyError
        self._tracker = UniTrackTracker(cfg["cfg_path"]["unitrack"], logger)

    def __del__(self):
        del self._detector, self._tracker, self._logger

    def predict(self, video_path: str, data_dir: str):
        # create video capture
        video_capture = Capture(video_path)
        assert (
            video_capture.is_opened
        ), f"{video_path} does not exist or is wrong file type."

        # create video writer
        out_path = os.path.join(data_dir, "video", "keypoints.mp4")
        video_writer = Writer(out_path, video_capture.fps, video_capture.size)

        json_data = []
        self._logger.info(f"=> loading video from {video_path}.")
        data_loader = make_test_dataloader(video_capture)
        self._logger.info(f"=> writing video into {out_path} while processing.")
        for frame_num, imgs in enumerate(tqdm(data_loader)):
            frame_num += 1  # frame_num = (1, ...)
            assert 1 == imgs.size(0), "Test batch size should be 1"
            frame = imgs[0].cpu().numpy()

            # do keypoints detection and tracking
            kps = self._detector.predict(frame)
            tracks = self._tracker.update(frame, kps)

            # write video
            self._write_video(video_writer, frame, tracks, frame_num)

            # append result
            for t in tracks:
                data = {
                    "frame": frame_num,
                    "id": t.track_id,
                    "keypoints": np.array(t.pose),
                }
                json_data.append(data)
                del data  # release memory

            del imgs, frame, kps, tracks  # release memory

        json_path = os.path.join(data_dir, ".json", "keypoints.json")
        self._logger.info(f"=> writing json file into {json_path}.")
        dump(json_data, json_path)

        # release memory
        del (video_capture, video_writer, data_loader, json_data)

    @staticmethod
    def _write_video(
        writer: Writer, frame: NDArray, tracks: List[STrack], frame_num: int
    ):
        # add keypoints to image
        frame = put_frame_num(frame, frame_num)
        for t in tracks:
            frame = draw_skeleton(frame, t.track_id, t.pose)

        writer.write(frame)
