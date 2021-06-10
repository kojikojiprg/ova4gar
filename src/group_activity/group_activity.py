from common import common
from common.json import GA_FORMAT
from group_activity.indicator import INDICATOR_DICT
from group_activity.passing_detector import PassingDetector
import os


class GroupActivity:
    def __init__(self, homo, field, method=None):
        self.homo = homo
        self.field = field
        self.method = method
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}

        self.pass_clf = PassingDetector(os.path.join(common.model_dir, 'pass_model.pickle'))

    def calc_indicator(self, frame_num, individual_activity_datas, **karg):
        if self.method is None:
            for key, func in INDICATOR_DICT.items():
                if key == list(GA_FORMAT.keys())[0]:
                    # key == attention
                    angle = karg['angle_range']
                    self.indicator_dict[key] += func(
                        frame_num, individual_activity_datas, self.homo, self.field, angle_range=angle)
                elif key == list(GA_FORMAT.keys())[1]:
                    # key == passing
                    self.indicator_dict[key] += func(
                        frame_num, individual_activity_datas, self.homo, self.pass_clf)
                else:
                    self.indicator_dict[key] += func(
                        frame_num, individual_activity_datas, self.homo)
        else:
            func = INDICATOR_DICT[self.method]
            if self.method == list(GA_FORMAT.keys())[0]:
                # method == attention
                angle = karg['angle_range']
                self.indicator_dict[self.method] += func(
                    frame_num, individual_activity_datas, self.homo, self.field, angle_range=angle)
            elif self.method == list(GA_FORMAT.keys())[1]:
                # key == passing
                self.indicator_dict[self.method] += func(
                    frame_num, individual_activity_datas, self.homo, self.pass_clf)
            else:
                self.indicator_dict[self.method] += func(
                    frame_num, individual_activity_datas, self.homo)

    def to_json(self):
        return self.indicator_dict
