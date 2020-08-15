import numpy as np
from person import Person


class Tracker:
    def __init__(self, keypoints_frame_lst):
        self.keypoints_frame_lst = keypoints_frame_lst
        self.persons = []

        keypoints_lst = self.keypoints_frame_lst[0]
        for i, keypoints in enumerate(keypoints_lst):
            self.persons.append(Person(i, keypoints))

    def track(self):
        for keypoints_lst in self.keypoints_frame_lst:
            targets = keypoints_lst.get_middle_points('Hip')

            for person in self.persons:
                keypoints = None
                nearest = np.inf
                for i, target in enumerate(targets):
                    # マハラノビス距離を求める
                    distance = person.pf.mahalanobis(target)

                    # 一番距離が近いポイントを取り出す
                    if distance < nearest:
                        keypoints = keypoints_lst[i]
                        nearest = distance

                # パーティクルフィルタを更新
                person.update(keypoints)

        return self.persons
