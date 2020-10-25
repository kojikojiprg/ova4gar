from group.indicator import INDICATOR_DICT


class Group:
    def __init__(self, homo):
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}

        self.homo = homo

    def append_calc(self, person_datas):
        for k in self.indicator_dict.keys():
            self.indicator_dict[k].append(INDICATOR_DICT[k](person_datas, self.homo))

    def append_data(self, data):
        for i, k in enumerate(self.indicator_dict.keys()):
            self.indicator_dict[k].append(data[i + 1])

    def get_data(self, frame_num):
        data = [frame_num]
        for k in self.indicator_dict.keys():
            data.append(self.indicator_dict[k][frame_num])

        return data
