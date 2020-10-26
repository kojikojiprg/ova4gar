from group.indicator import INDICATOR_DICT
from group.display import DISPLAY_DICT, HEATMAP_SETTING_DICT
from display.heatmap import Heatmap


class Group:
    def __init__(self, homo):
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}
        self.heatmap_dict = {}

        self.homo = homo

    def append_calc(self, k, frame_num, person_datas):
        self.indicator_dict[k] += INDICATOR_DICT[k](
            frame_num, person_datas, self.homo)

    def append_data(self, k, datas):
        for data in datas:
            self.indicator_dict[k].append(data)

    def get_data(self, k, frame_num):
        data = []
        for row in self.indicator_dict[k]:
            if int(row[0]) == frame_num:
                data.append(row)

        return data

    def make_heatmap(self):
        for k in self.indicator_dict.keys():
            if HEATMAP_SETTING_DICT[k][0]:
                # ヒートマップを作成する場合
                distribution = []
                for data in self.indicator_dict[k]:
                    # ヒータマップの対象となる列を取得
                    data_idx = HEATMAP_SETTING_DICT[k][1]
                    distribution.append(data[data_idx])
                # ヒートマップ作成
                self.heatmap_dict[k] = Heatmap(distribution)
            else:
                self.heatmap_dict[k] = None

    def display(self, k, frame_num, field):
        if HEATMAP_SETTING_DICT[k][0]:
            field = DISPLAY_DICT[k](
                self.get_data(k, frame_num), field, self.heatmap_dict[k])
        else:
            field = DISPLAY_DICT[k](
                self.get_data(k, frame_num), field)

        return field
