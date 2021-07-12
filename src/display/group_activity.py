from common.json import GA_FORMAT
from display.heatmap import Heatmap
import inspect
import cv2


keys = list(GA_FORMAT.keys())
HEATMAP_SETTING_DICT = {
    # key: [is_heatmap, heatmap_data_index]
    keys[0]: [False, None],
    keys[1]: [False, None],
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

    def disp_attention(self, datas, field):
        key = inspect.currentframe().f_code.co_name.replace('disp_', '')
        json_format = GA_FORMAT[key]

        for data in datas:
            point = data[json_format[1]]
            person_points = data[json_format[2]]
            # likelihood = data[json_format[3]]

            # color = self.heatmap_dict[key].colormap(likelihood)
            for person_point in person_points:
                cv2.line(field, point, person_point, color=(255, 165, 0), thickness=2)

            cv2.circle(field, tuple(point), 8, (255, 165, 0), thickness=-1)

        return field

    def disp_passing(self, datas, field, persons=None):
        key = inspect.currentframe().f_code.co_name.replace('disp_', '')
        json_format = GA_FORMAT[key]

        for data in datas:
            is_persons = False
            if persons is None:
                is_persons = True
            else:
                is_persons = data[json_format[1]][0] in persons and data[json_format[1]][1] in persons

            points = data[json_format[2]]
            pred = data[json_format[3]]
            if is_persons and points is not None and pred == 1:
                cv2.line(field, points[0], points[1], color=(255, 165, 0), thickness=2)

        return field
