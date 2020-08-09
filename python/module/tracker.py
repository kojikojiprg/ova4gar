import numpy as np
from person import Person


class Tracker:
    def __init__(self, keypoints_frame_lst):
        self.keypoints_frame_lst = keypoints_frame_lst

        keypoints_lst = keypoints_frame_lst[0]

        self.persons = []
        for i, keypoints in enumerate(keypoints_lst):
            self.persons.append(Person(i, keypoints))
        self.persons = np.array(self.persons)

    def track_person(self, person_id):
        person = self.persons[person_id]
        for keypoints_lst in self.keypoints_frame_lst:
            targets = keypoints_lst.get_middle_points('Ankle')

            point = None
            keypoints = None
            nearest = np.inf
            for i, target in enumerate(targets):
                # マハラノビス距離を求める
                distance = person.pf.mahalanobis(target)

                # 一番距離が近いポイントを取り出す
                if distance < nearest:
                    point = target
                    keypoints = keypoints_lst[i]
                    nearest = distance

            # パーティクルフィルタを更新
            person.update(point, keypoints)

        return person

    def track(self, person_id_lst):
        persons = self.persons[person_id_lst]
        # WIP
