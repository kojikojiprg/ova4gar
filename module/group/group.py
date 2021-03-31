from group.indicator import INDICATOR_DICT


class Group:
    def __init__(self, homo):
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}
        self.heatmap_dict = {}

        self.homo = homo

    def calc_indicator(self, frame_num, person_datas):
        for key, func in INDICATOR_DICT.items():
            self.indicator_dict[key] += func(
                frame_num, person_datas, self.homo)

    def to_json(self):
        return self.indicator_dict
