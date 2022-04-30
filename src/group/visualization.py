from typing import Any, Dict, List, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from group.heatmap import Heatmap

HEATMAP_SETTING = {
    # key: [is_heatmap, min, max]
    "passing": (False, None, None),
    "attention": (True, 0, 2),
}


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

    def visualize(
        self,
        key: str,
        frame_num: int,
        group_indicator_data: Dict[str, List[Dict[str, Any]]],
        field: NDArray,
    ) -> NDArray:
        data = group_indicator_data[key]

        # フレームごとにデータを取得する
        data_per_frame = [item for item in data if item["frame"] == frame_num]

        # 指標を書き込む
        field = eval(f"self._{key}")(data_per_frame, field)

        return field

    def _passing(self, data, field, persons=None, alpha=0.2):
        for item in data:
            is_persons = False
            if persons is None:
                is_persons = True
            else:
                is_persons = (
                    item["persons"][0] in persons and item["persons"][1] in persons
                )

            points = item["points"]
            pred = item["pred"]
            if is_persons and points is not None and pred == 1:
                p1 = np.array(points[0])
                p2 = np.array(points[1])

                # 楕円を計算
                diff = p2 - p1
                center = p1 + diff / 2
                major = int(np.abs(np.linalg.norm(diff))) + 20
                minor = int(major * 0.5)
                angle = np.rad2deg(np.arctan2(diff[1], diff[0]))

                # 描画
                cv2.line(field, p1, p2, color=(255, 165, 0), thickness=1)
                copy = field.copy()
                cv2.ellipse(
                    copy,
                    (center, (major, minor), angle),
                    color=(200, 200, 255),
                    thickness=-1,
                )
                field = cv2.addWeighted(copy, alpha, field, 1 - alpha, 0)

        return field

    def _attention(self, data, field, alpha=0.2, max_radius=15):
        copy = field.copy()
        for item in data:
            point = item["point"]
            value = item["value"]

            color = self._heatmaps["attention"].colormap(value)

            # calc radius of circle
            max_value = self._heatmaps["attention"].xmax
            radius = int(value / max_value * max_radius)
            if radius == 0:
                radius = 1

            cv2.circle(copy, tuple(point), radius, color, thickness=-1)

        field = cv2.addWeighted(copy, alpha, field, 1 - alpha, 0)

        return field
