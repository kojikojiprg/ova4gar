from group.indicator import INDICATOR_DICT
from group.display import DISPLAY_DICT


class Group:
    def __init__(self, homo):
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}

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

    def display(self, k, frame_num, field):
        field = DISPLAY_DICT[k](
            self.get_data(k, frame_num), field)

        return field
