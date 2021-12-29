import inspect

import cv2
import numpy as np
from common.default import ATTENTION_DEFAULT
from common.json import GA_FORMAT

from display.heatmap import Heatmap

keys = list(GA_FORMAT.keys())
HEATMAP_SETTING_DICT = {
    # key: [is_heatmap, heatmap_data_index, min, max]
    keys[0]: (True, None, 0, 2),
    keys[1]: (False, None, None, None),
}


class DisplayGroupActivity:
    def __init__(self, group_activity_datas):
        self.heatmap_dict = {}
        self.make_heatmap(group_activity_datas)

    def make_heatmap(self, group_activity_datas):
        for key, datas in group_activity_datas.items():
            if HEATMAP_SETTING_DICT[key][0]:
                # ヒートマップを作成する場合
                distribution = []
                if HEATMAP_SETTING_DICT[key][2] is None:
                    # ヒートマップをデータから作成
                    data_keys = GA_FORMAT[key]
                    for data in datas:
                        append_data = data[data_keys[HEATMAP_SETTING_DICT[key][1]]]
                        if append_data is not None:
                            distribution.append(append_data)
                    print(f'max of {key}: ', np.max(distribution))
                else:
                    # ヒートマップをminとmaxから作成
                    distribution = [
                        HEATMAP_SETTING_DICT[key][2],
                        HEATMAP_SETTING_DICT[key][3],
                    ]

                if len(distribution) > 0:
                    self.heatmap_dict[key] = Heatmap(distribution)
                else:
                    self.heatmap_dict[key] = None
            else:
                self.heatmap_dict[key] = None

    def disp(self, key, frame_num, group_activity_datas, field):
        indicator_datas = group_activity_datas[key]

        # フレームごとにデータを取得する
        frame_indicator_datas = [
            data for data in indicator_datas if data["frame"] == frame_num
        ]

        # 指標を書き込む
        field = eval("self.disp_{}".format(key))(frame_indicator_datas, field)

        return field

    def disp_attention(self, datas, field, alpha=0.2, max_radius=15):
        key = inspect.currentframe().f_code.co_name.replace("disp_", "")
        json_format = GA_FORMAT[key]

        copy = field.copy()
        for data in datas:
            point = data[json_format[2]]
            value = data[json_format[4]]

            color = self.heatmap_dict[key].colormap(value)

            # calc radius of circle
            max_value = self.heatmap_dict[key].xmax
            radius = int(value / max_value * max_radius)
            if radius == 0:
                radius = 1

            cv2.circle(copy, tuple(point), radius, color, thickness=-1)

        field = cv2.addWeighted(copy, alpha, field, 1 - alpha, 0)

        return field

    # def disp_attention(self, datas, field, th=2):
    #     key = inspect.currentframe().f_code.co_name.replace("disp_", "")
    #     json_format = GA_FORMAT[key]

    #     for data in datas:
    #         point = data[json_format[2]]
    #         count = data[json_format[4]]
    #         if count >= th:
    #             cv2.circle(field, tuple(point), 10, (255, 165, 0), thickness=-1)
    #             cv2.circle(field, tuple(point), 45, (255, 165, 0), thickness=3)

    #     return field

    def disp_passing(self, datas, field, persons=None, alpha=0.2):
        key = inspect.currentframe().f_code.co_name.replace("disp_", "")
        json_format = GA_FORMAT[key]

        for data in datas:
            is_persons = False
            if persons is None:
                is_persons = True
            else:
                is_persons = (
                    data[json_format[1]][0] in persons
                    and data[json_format[1]][1] in persons
                )

            points = data[json_format[2]]
            pred = data[json_format[3]]
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
