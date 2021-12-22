import os

from common import common
from common.json import GA_FORMAT

from group_activity.indicator import INDICATOR_DICT
from group_activity.passing_detector_lstm import PassingDetector


class GroupActivity:
    def __init__(self, field, method=None):
        self.field = field
        self.method = method
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}
        self.queue_dict = {k: {} for k in INDICATOR_DICT.keys()}

        self.pass_clf = PassingDetector(
            os.path.join(common.model_dir, "config/pass_model_lstm.yaml"),
            os.path.join(common.model_dir, "checkpoint/pass_model_lstm.pth")
        )

    def calc_indicator(self, frame_num, individual_activity_datas):
        for key, func in INDICATOR_DICT.items():
            if self.method is not None and self.method != key:
                continue

            if key == list(GA_FORMAT.keys())[0]:
                # key == attention
                value, queue = func(frame_num, individual_activity_datas, self.queue_dict[key], self.field)
            elif key == list(GA_FORMAT.keys())[1]:
                # key == passing
                value, queue = func(frame_num, individual_activity_datas, self.queue_dict[key], self.pass_clf)
            else:
                value, queue = func(frame_num, individual_activity_datas, self.queue_dict[key])

            self.indicator_dict[key] += value
            self.queue_dict[key] = queue

    def to_json(self):
        return self.indicator_dict
