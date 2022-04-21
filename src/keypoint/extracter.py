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
        self.logger = logger
        self.detector = HRNetDetecter(hrnet_cfg_path, hrnet_opts, logger)
        self.tracker = UniTrackTracker(unitrack_opts, logger)

    def __del__(self):
        del self.detector, self.tracker, self.logger

    def predict(self, video_path: str, data_dir: str):
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
        for frame_num, (imgs, img_raws) in enumerate(data_loader):

            assert 1 == imgs.size(0), "Test batch size should be 1"
            img = imgs[0].cpu().numpy()
            img_raw = img_raws[0].cpu().numpy()

            # do keypoints detection and tracking
            keypoints = self.detector.predict(img_raw)
            tracks = self.tracker.update(img, img_raw, keypoints)

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

            del imgs, img_raws, img, img_raw, keypoints, tracks  # release memory

            pbar.update()

        pbar.close()

        self.logger.info(f"=> writing json file into {json_path}.")
        self._write_json(json_data, json_path)

        # release memory
        del (video_capture, video_writer, data_loader, test_dataset, json_data)

    def _write_video(self, writer: Writer, image, tracks):
        # add keypoints to image
        for t in tracks:
            draw_skeleton(image, t.track_id, t.pose)

        image = image.astype(np.uint8)
        writer.write(image)

    def _write_json(self, json_data, json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        dump(json_data, json_path)
