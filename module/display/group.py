from common.json import GROUP_FORMAT
from display.heatmap import Heatmap
import inspect
import numpy as np
import cv2


keys = list(GROUP_FORMAT.keys())
HEATMAP_SETTING_DICT = {
    # key: [is_heatmap, heatmap_data_index]
    keys[0]: [True, -1],
    keys[1]: [True, -1],
}


class DisplayGroup:
    def __init__(self, group_datas):
        self.heatmap_dict = {}
        self.make_heatmap(group_datas)

    def make_heatmap(self, group_datas):
        for key, datas in group_datas.items():
            if HEATMAP_SETTING_DICT[key][0]:
                # ヒートマップを作成する場合
                distribution = []
                data_keys = GROUP_FORMAT[key]
                for data in datas:
                    distribution.append(
                        data[data_keys[HEATMAP_SETTING_DICT[key][1]]])
                self.heatmap_dict[key] = Heatmap(distribution)
            else:
                self.heatmap_dict[key] = None

    def disp(self, key, frame_num, group_datas, field):
        indicator_datas = group_datas[key]

        # フレームごとにデータを取得する
        frame_indicator_datas = [
            data for data in indicator_datas if data['image_id'] == frame_num]

        # 指標を書き込む
        field = eval('self.disp_{}'.format(key))(
            frame_indicator_datas, field)

        return field

    def disp_density(self, datas, field, min_r=8):
        key = inspect.currentframe().f_code.co_name.replace('disp_', '')
        json_format = GROUP_FORMAT[key]

        for data in datas:
            points = data[json_format[1]]
            point = np.average(points, axis=0).astype(int)
            r = min_r + len(points)
            color = self.heatmap_dict[key].colormap(len(points))
            cv2.circle(field, tuple(point), r, color, thickness=-1)

        return field

    def disp_attention(self, datas, field, min_r=8):
        key = inspect.currentframe().f_code.co_name.replace('disp_', '')
        json_format = GROUP_FORMAT[key]

        for data in datas:
            points = np.array(data[json_format[1]])
            for point in points:
                # 視線が重なったところを黒色で表示
                cv2.circle(field, tuple(point.astype(int)), 5, (0, 0, 0), thickness=-1)

            # クラスターを表示
            point = np.average(points, axis=0).astype(int)
            r = min_r + len(points)
            color = self.heatmap_dict[key].colormap(len(points))
            cv2.circle(field, tuple(point), r, color, thickness=-1)

        return field

    def disp_passing(self, datas, field):
        key = inspect.currentframe().f_code.co_name.replace('disp_', '')
        json_format = GROUP_FORMAT[key]

        for data in datas:
            point = data[json_format[1]]
            likelifood = data[json_format[0]]
            if point is not None:
                cv2.circle(field, tuple(point), 5, (0, 0, 0), thickness=-1)
                cv2.putText(
                    field, str(likelifood), tuple(point), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
