import gc
import os
import sys
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

sys.path.append("src")
from utility.json_handler import load
from utility.video import Capture, Writer, concat_field_with_frame

from visualize.heatmap import Heatmap
from visualize.individual import write_field as ind_write_field
from visualize.keypoint import write_frame as kps_write_frame

HEATMAP_SETTING = {
    # key: [is_heatmap, min, max]
    "passing": (False, None, None),
    "attention": (True, 0, 2),
}


def write_video(data_dir, keys, field, start_frame_num, end_frame_num):
    # create video capture
    video_path = os.path.join(data_dir, "video", "keypoints.mp4")
    cap = Capture(video_path)

    # create video writer
    cmb_img = concat_field_with_frame(cap.read()[1], field)
    size = cmb_img.shape[1::-1]
    writers: Dict[str, Writer] = {}
    out_paths = []
    for key in keys:
        out_path = os.path.join(data_dir, "video", f"{key}.mp4")
        out_paths.append(out_path)
        writers[key] = Writer(out_path, cap.fps, size)

    # load json file
    json_path = os.path.join(data_dir, ".json", "keypoints.json")
    kps_data = load(json_path)
    json_path = os.path.join(data_dir, ".json", "individual.json")
    ind_data = load(json_path)
    json_path = os.path.join(data_dir, ".json", "group.json")
    grp_data = load(json_path)

    visualizer = GroupVisualizer(keys)

    cap.set_pos_frame_count(start_frame_num - 1)
    for frame_num in tqdm(range(start_frame_num, end_frame_num + 1)):
        _, frame = cap.read()
        frame = kps_write_frame(frame, kps_data, frame_num)
        field = ind_write_field(ind_data, field, frame_num)
        for key in keys:
            field_tmp = visualizer.write_field(key, frame_num, grp_data, field.copy())
            frame_tmp = concat_field_with_frame(frame.copy(), field_tmp)
            writers[key].write(frame_tmp)

    # release memory
    del cap, writers, kps_data, ind_data, grp_data
    gc.collect()


class GroupVisualizer:
    def __init__(self, keys: List[str]):
        self._heatmaps: Dict[str, Union[Heatmap, None]] = {}
        self._make_heatmap(keys)

    def _make_heatmap(self, keys: List[str]):
        for key in keys:
            if HEATMAP_SETTING[key][0]:
                # ヒートマップを作成する場合
                distribution = [
                    HEATMAP_SETTING[key][1],
                    HEATMAP_SETTING[key][2],
                ]
                self._heatmaps[key] = Heatmap(distribution)
            else:
                self._heatmaps[key] = None

    def write_field(
        self,
        key: str,
        frame_num: int,
        group_indicator_data: Dict[str, Dict[int, List[Dict[str, Any]]]],
        field: NDArray,
        alpha: float = 0.3,
    ) -> NDArray:
        if frame_num not in group_indicator_data[key]:
            return field

        data = group_indicator_data[key][frame_num]

        copy = field.copy()
        for item in data:
            if key == "passing":
                copy = self._passing(item, copy)
            elif key == "attention":
                copy = self._attention(item, copy)
            else:
                raise KeyError
        field = cv2.addWeighted(copy, alpha, field, 1 - alpha, 0)

        return field

    def _passing(self, item, field):
        points = item["points"]

        p1 = np.array(points[0])
        p2 = np.array(points[1])

        # 楕円を計算
        diff = p2 - p1
        center = p1 + diff / 2
        major = int(np.abs(np.linalg.norm(diff))) + 20
        minor = int(major * 0.5)
        angle = np.rad2deg(np.arctan2(diff[1], diff[0]))

        # 描画
        cv2.ellipse(
            field,
            (center, (major, minor), angle),
            color=(200, 200, 255),
            thickness=-1,
        )
        cv2.line(field, p1, p2, color=(185, 105, 0), thickness=3)

        return field

    def _attention(self, item, field, max_radius=15):
        point = item["point"]
        value = item["value"]

        color = self._heatmaps["attention"].colormap(value)

        # calc radius of circle
        max_value = self._heatmaps["attention"].xmax
        radius = int(value / max_value * max_radius)
        if radius == 0:
            return field

        cv2.circle(field, tuple(point), radius, color, thickness=-1)

        return field
