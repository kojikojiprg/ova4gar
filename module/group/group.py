from group.indicator import INDICATOR_DICT


class Group:
    def __init__(self, homo, field, method=None):
        self.homo = homo
        self.field = field
        self.method = method
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}

    def calc_indicator(self, frame_num, person_datas):
        if self.method is None:
            for key, func in INDICATOR_DICT.items():
                if key == 'attention':
                    self.indicator_dict[key] += func(
                        frame_num, person_datas, self.homo, self.field)
                else:
                    self.indicator_dict[key] += func(
                        frame_num, person_datas, self.homo)
        else:
            func = INDICATOR_DICT[self.method]
            if self.method == 'attention':
                self.indicator_dict[self.method] += func(
                    frame_num, person_datas, self.homo, self.field)
            else:
                self.indicator_dict[self.method] += func(
                    frame_num, person_datas, self.homo)

    def to_json(self):
        return self.indicator_dict
