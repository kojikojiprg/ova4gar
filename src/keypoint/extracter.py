import gc
import os
from logging import Logger
from typing import List

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from unitrack.tracker.mot.basetrack import STrack
from utility.json_handler import dump
from utility.video import Capture, Writer

from keypoint.detector.dataset import make_data_loader
from keypoint.detector.higher_hrnet import HigherHRNetDetecter
from keypoint.detector.hrnet import HRNetDetecter
from keypoint.tracker.unitrack import UniTrackTracker
from keypoint.visualization import draw_skeleton, put_frame_num


class Extractor:
    def __init__(self, keypoint_cfg: dict, logger: Logger, device: str):
        self._logger = logger

        self._cfg = keypoint_cfg
        detector_name = self._cfg["detector"]
        if detector_name == "higher_hrnet":
            self._detector = HigherHRNetDetecter(
                self._cfg["cfg_path"]["higher_hrnet"], logger, device
            )
        elif detector_name == "hrnet":
            self._detector = HRNetDetecter(
                self._cfg["cfg_path"]["hrnet"], logger, device
            )
        else:
            raise KeyError

        self._tracker = UniTrackTracker(
            self._cfg["cfg_path"]["unitrack"], logger, device
        )

        self._batch_size = self._cfg["batch_size"]

    def __del__(self):
        torch.cuda.empty_cache()
        del self._detector, self._tracker, self._logger
        gc.collect()

    def predict(self, video_path: str, data_dir: str, writing_video: bool = False):
        if writing_video:
            # create video capture
            self._logger.info(f"=> loading video from {video_path}.")
            video_capture = Capture(video_path)
            assert (
                video_capture.is_opened
            ), f"{video_path} does not exist or is wrong file type."

            # create video writer
            out_path = os.path.join(data_dir, "video", "keypoints.mp4")
            video_writer = Writer(out_path, video_capture.fps, video_capture.size)

        kps_all = self._detect(video_capture)

        self._logger.info("=> tracking keypoints")
        if writing_video:
            self._logger.info(f"=> writing video into {out_path} while processing.")
            video_capture.set_pos_frame_count(0)  # initialize video capture
        json_data = []
        for frame_num, kps in enumerate(tqdm(kps_all)):
            frame_num += 1  # frame_num = (1, ...)
            _, frame = video_capture.read()

            # tracking
            tracks = self._tracker.update(frame, kps)

            if writing_video:
                self.write_video(video_writer, frame, tracks, frame_num)

            # append result
            for t in tracks:
                data = {
                    "frame": frame_num,
                    "id": t.track_id,
                    "keypoints": np.array(t.pose),
                }
                json_data.append(data)

            del tracks
            gc.collect()

        json_path = os.path.join(data_dir, ".json", "keypoints.json")
        self._logger.info(f"=> writing json file into {json_path}.")
        dump(json_data, json_path)

        # release memory
        torch.cuda.empty_cache()
        del kps_all, json_data
        if writing_video:
            del video_capture, video_writer
        gc.collect()

    def _detect(self, video_capture):
        self._logger.info("=> detecting keypoints")
        dataloader = make_data_loader(video_capture, self._batch_size)
        kps_all = []
        for frame_nums, frames in tqdm(dataloader):
            frames = [frame.cpu().numpy() for frame in frames]

            # keypoints detection
            kps_all_batch = self._detector.predict(frames)
            for kps in kps_all_batch:
                kps = self._del_leaky(kps, self._cfg["th_delete"])
                kps = self._get_unique(kps, self._cfg["th_diff"], self._cfg["th_count"])
                kps_all.append(kps)

        return kps_all

    @staticmethod
    def _del_leaky(kps: NDArray, th_delete: float):
        return kps[np.nonzero(np.mean(kps[:, :, 2], axis=1) > th_delete)]

    @staticmethod
    def _get_unique(kps: NDArray, th_diff: float, th_count: int):
        unique_kps = np.empty((0, 17, 3))

        for i in range(len(kps)):
            found_overlap = False

            for j in range(len(unique_kps)):
                diff = np.linalg.norm(kps[i, :, :2] - unique_kps[j, :, :2], axis=1)
                if len(np.where(diff < th_diff)[0]) >= th_count:
                    found_overlap = True
                    if np.mean(kps[i, :, 2]) > np.mean(unique_kps[j, :, 2]):
                        # select one has more confidence score
                        unique_kps[j] = kps[i]
                    break

            if not found_overlap:
                # if there aren't overlapped
                unique_kps = np.append(unique_kps, [kps[i]], axis=0)

        return unique_kps

    @staticmethod
    def write_video(
        writer: Writer, frame: NDArray, tracks: List[STrack], frame_num: int
    ):
        # add keypoints to image
        frame = put_frame_num(frame, frame_num)
        for t in tracks:
            frame = draw_skeleton(frame, t.track_id, np.array(t.pose))

        writer.write(frame)
