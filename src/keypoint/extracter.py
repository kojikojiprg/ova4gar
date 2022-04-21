import os
from logging import Logger

import numpy as np
from tqdm import tqdm

from utility.json_handler import dump
from utility.video import Capture, Writer
from utility.visualization import draw_skeleton

from .dataset import make_test_dataloader
from .hrnet import HRNetDetecter
from .unitrack import UniTrackTracker


class Extractor:
    def __init__(self, hrnet_cfg_path: str, hrnet_opts: list, unitrack_opts, logger: Logger):
        self.detector = HRNetDetecter(hrnet_cfg_path, hrnet_opts, logger)
        self.tracker = UniTrackTracker(unitrack_opts)
        self.logger = logger

    def __del__(self):
        del self.detector, self.logger

    def extract(self, video_path: str, data_dir: str):
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

            # do keypoints detection and tracking
            keypoints = self.detector.predict(image)
            tracks = self.tracker.update(image, keypoints)

            self._write_video(video_writer, image, tracks)  # write video

            # append result
            for t in tracks:
                data = {
                    "frame": frame_num + 1,
                    "id": t.track_id,
                    "keypoints": np.array(t.pose),
                }
                json_data.append(data)

            pbar.update()

        pbar.close()

        self.logger.info(f" => writing json file into {json_path}.")
        self._write_json(json_data, json_path)

        # release memory
        del (video_capture, video_writer, data_loader, test_dataset, json_data, data)

    def _write_video(self, writer: Writer, image, tracks):
        # add keypoints to image
        for t in tracks:
            draw_skeleton(image, t.t_id, t.pose)

        writer.write(image)

    def _write_json(self, json_data, json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        dump(json_data, json_path)
