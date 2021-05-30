from common.json import GA_FORMAT
from display.heatmap import Heatmap
import inspect
import numpy as np
import cv2


keys = list(GA_FORMAT.keys())
HEATMAP_SETTING_DICT = {
    # key: [is_heatmap, heatmap_data_index]
    keys[0]: [True, -1],
    keys[1]: [True, -1],
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
                data_keys = GA_FORMAT[key]
                for data in datas:
                    append_data = data[data_keys[HEATMAP_SETTING_DICT[key][1]]]
                    if append_data is not None:
                        distribution.append(append_data)

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
            data for data in indicator_datas if data['frame'] == frame_num]

        # 指標を書き込む
        field = eval('self.disp_{}'.format(key))(
            frame_indicator_datas, field)

        return field

    def disp_density(self, datas, field, min_r=8):
        key = inspect.currentframe().f_code.co_name.replace('disp_', '')
        json_format = GA_FORMAT[key]

        for data in datas:
            points = data[json_format[1]]
            point = np.average(points, axis=0).astype(int)
            r = min_r + len(points)
            color = self.heatmap_dict[key].colormap(len(points))
            cv2.circle(field, tuple(point), r, color, thickness=-1)

        return field

    def disp_attention(self, datas, field):
        key = inspect.currentframe().f_code.co_name.replace('disp_', '')
        json_format = GA_FORMAT[key]

        for data in datas:
            point = data[json_format[1]]
            value = data[json_format[2]]

            color = self.heatmap_dict[key].colormap(value)
            cv2.circle(field, tuple(point), 1, color, thickness=-1)

        return field

    def disp_passing(self, datas, field, persons=None):
        key = inspect.currentframe().f_code.co_name.replace('disp_', '')
        json_format = GA_FORMAT[key]

        for data in datas:
            is_persons = False
            if persons is None:
                is_persons = True
            else:
                is_persons = data[json_format[2]][0] in persons and data[json_format[2]][1] in persons

            point = data[json_format[1]]
            if point is not None:
                if is_persons:
                    likelifood = np.round(data[json_format[3]], decimals=3)
                    cv2.circle(field, tuple(point), 5, (0, 255, 0), thickness=-1)
                    cv2.putText(field, str(likelifood), tuple(point),
                                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        return field
