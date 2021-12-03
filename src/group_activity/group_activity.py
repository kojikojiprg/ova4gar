import os

from common import common
from common.json_io_io import GA_FORMAT

from group_activity.indicator import INDICATOR_DICT
from group_activity.passing_detector import PassingDetector


class GroupActivity:
    def __init__(self, field, method=None):
        self.field = field
        self.method = method
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}

        self.pass_clf = PassingDetector(
            os.path.join(common.model_dir, "pass_model_svm.pickle")
        )

    def calc_indicator(self, frame_num, individual_activity_datas):
        if self.method is None:
            for key, func in INDICATOR_DICT.items():
                # if key == list(GA_FORMAT.keys())[0]:
                #     # key == attention
                #     self.indicator_dict[key] += func(
                #         frame_num, individual_activity_datas, self.field)
                # elif key == list(GA_FORMAT.keys())[1]:
                if key == list(GA_FORMAT.keys())[1]:
                    # key == passing
                    self.indicator_dict[key] += func(
                        frame_num, individual_activity_datas, self.pass_clf
                    )
                else:
                    self.indicator_dict[key] += func(
                        frame_num, individual_activity_datas
                    )
        else:
            func = INDICATOR_DICT[self.method]
            # if self.method == list(GA_FORMAT.keys())[0]:
            #     # method == attention
            #     self.indicator_dict[self.method] += func(
            #         frame_num, individual_activity_datas, self.field)
            # elif self.method == list(GA_FORMAT.keys())[1]:
            if self.method == list(GA_FORMAT.keys())[1]:
                # key == passing
                self.indicator_dict[self.method] += func(
                    frame_num, individual_activity_datas, self.pass_clf
                )
            else:
                self.indicator_dict[self.method] += func(
                    frame_num, individual_activity_datas
                )

    def to_json(self):
        return self.indicator_dict
