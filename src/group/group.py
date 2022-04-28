import os
from logging import Logger
from typing import Any, Dict, List

import numpy as np
from individual.individual import Individual
from tqdm import tqdm
from utility.video import (
    Capture,
    Writer,
    concat_field_with_frame,
    divide_field_and_frame,
)

from group.indicator import attention, passing
from group.passing_detector import PassingDetector
from group.visualization import GroupVisualizer


class Group:
    def __init__(self, cfg: dict, field: np.typing.NDArray, logger: Logger):
        self._keys = list(cfg["indicator"].keys())
        self._funcs = {k: eval(k) for k in self._keys}
        self._defs: Dict[str, Any] = {}
        for ind_key, item in cfg["indicator"].items():
            self._defs[ind_key] = {}
            for key, val in item["default"].items():
                self._defs[ind_key][key] = val

        self._field = field
        self._logger = logger

        pass_cfg_path = cfg["indicator"]["passing"]["cfg_path"]
        self._logger.info(f"=> load passing detector from {pass_cfg_path}")
        self._pass_clf = PassingDetector(pass_cfg_path, self._defs["passing"])
        self._pass_clf.eval()

        self._idc_dict: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self._keys}
        self._idc_que: Dict[str, Any] = {
            "attention": [],
            "passing": {},
        }

        self._max_frame_num = 0

    def __del__(self):
        del self._field, self._logger
        del self._pass_clf
        del self._idc_dict, self._idc_que

    def calc_indicator(self, frame_num: int, individuals: List[Individual]):
        for key, func in self._funcs.items():
            if key == "passing":
                value, queue = func(
                    frame_num,
                    individuals,
                    self._idc_que[key],
                    self._pass_clf,
                )
            elif key == "attention":
                value, queue = func(
                    frame_num,
                    individuals,
                    self._idc_que[key],
                    self._field,
                    self._defs["attention"],
                )
            else:
                raise KeyError

            if value is not None:
                self._idc_dict[key] += value
            self._idc_que[key] = queue
        self._max_frame_num = max(self._max_frame_num, frame_num)

    def to_json(self):
        return self._idc_dict

    def write_video(self, data_dir: str):
        # create visualizer
        visualizer = GroupVisualizer(self._idc_dict)

        # create video capture
        video_path = os.path.join(data_dir, "video", "individual.mp4")
        self._logger.info(f"=> loading video from {video_path}.")
        video_capture = Capture(video_path)
        assert (
            video_capture.is_opened
        ), f"{video_path} does not exist or is wrong file type."

        # create video writer
        writers: Dict[str, Writer] = {}
        for key in self._keys:
            out_path = os.path.join(data_dir, "video", f"{key}.mp4")
            video_writer = Writer(out_path, video_capture.fps, video_capture.size)
            writers[key] = video_writer

        self._logger.info(f"=> writing video into {out_path} while processing")
        for frame_num in tqdm(range(1, self._max_frame_num + 2)):
            # get frame and field
            _, frame = video_capture.read()
            frame, field = divide_field_and_frame(frame, self._field.shape[1])

            for key in self._keys:
                field_tmp = visualizer.visualize(
                    key, frame_num, self._idc_dict, field.copy()
                )
                frame_tmp = concat_field_with_frame(frame.copy(), field_tmp)
                writers[key].write(frame_tmp)

        del video_capture, writers  # release memory
