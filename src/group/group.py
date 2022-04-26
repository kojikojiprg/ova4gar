import os

from group.indicator import INDICATOR_DICT
from group.passing_detector import PassingDetector


class Group:
    def __init__(self, field):
        self.field = field
        self.indicator_dict = {k: [] for k in INDICATOR_DICT.keys()}
        self.queue_dict = {
            "attention": [],
            "passing": {},
        }

        self.pass_clf = PassingDetector(
            os.path.join(common.model_dir, ""),
            os.path.join(common.model_dir, "checkpoint/pass_model_lstm.pth"),
        )
        self.pass_clf.eval()

    def calc_indicator(self, frame_num, individual_activity_datas):
        for key, func in INDICATOR_DICT.items():
            if key == list(GA_FORMAT.keys())[0]:
                # key == attention
                value, queue = func(
                    frame_num,
                    individual_activity_datas,
                    self.queue_dict[key],
                    self.field,
                )
            elif key == list(GA_FORMAT.keys())[1]:
                # key == passing
                value, queue = func(
                    frame_num,
                    individual_activity_datas,
                    self.queue_dict[key],
                    self.pass_clf,
                )
            else:
                value, queue = func(
                    frame_num, individual_activity_datas, self.queue_dict[key]
                )

            self.indicator_dict[key] += value
            self.queue_dict[key] = queue

    def to_json(self):
        return self.indicator_dict
