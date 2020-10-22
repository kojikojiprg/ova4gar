from common import keypoint as kp
from common import database
from person.indicator import INDICATOR_DICT
import numpy as np


class Model:
    def __init__(self, person_id, start_frame_num):
        self.id = person_id
        self.start_frame_num = start_frame_num
        self.keypoints_lst = kp.KeypointsList()
        self.data_dict = {k: [] for k in INDICATOR_DICT.keys()}

    def append_calc(self, keypoints):
        if keypoints is None:
            return

        self.keypoints_lst.append(keypoints)
        for k in self.data_dict.keys():
            if keypoints.shape == (17, 3):
                keypoints_tmp = kp.Keypoints(keypoints)
                self.data_dict[k].append(INDICATOR_DICT[k](keypoints_tmp))
            else:
                self.data_dict[k].append(np.nan)

    def get_data(self, frame_num):
        idx = frame_num - self.start_frame_num
        if idx < 0:
            return None

        data = [self.id, frame_num, np.array(self.keypoints_lst[idx])]
        for k in self.data_dict.keys():
            data.append(self.data_dict[k][idx])

        return data


def make_person_database(tracking_db_path, person_db_path):
    tracking_db = database.DataBase(tracking_db_path)
    person_db = database.DataBase(person_db_path)

    tracking_datas = tracking_db.select(database.TRACKING_TABLE.name)

    models = []
    person_datas = []
    for row in tracking_datas:
        person_id = row[0]
        frame_num = row[1]
        keypoints = row[2]

        if len(models) == person_id:
            models.append(Model(person_id, frame_num))

        models[person_id].append_calc(keypoints)

        data = models[person_id].get_data(frame_num)
        if data is not None:
            person_datas.append(data)

    table = database.PERSON_TABLE
    person_db.drop_table(table.name)
    person_db.create_table(table.name, table.cols)
    person_db.insert_datas(
        table.name,
        list(table.cols.keys()),
        person_datas)
