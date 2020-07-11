import numpy as np
import particle_filter as pf


class Tracker:
    def __init__(self, keypoints_frame_lst):
        self.keypoints_frame_lst = keypoints_frame_lst

        keypoints_lst = keypoints_frame_lst[0]
        self.persons = keypoints_lst.get_middle_ankle_points()

    def _track(self, x, y, keypoint_lst):
        targets = keypoint_lst.get_middle_ankle_points()

        nearest = np.inf
        point = None
        for target in targets:
            # マハラノビス距離を求める
            distance = self.pf.mahalanobis(target)

            # 一番距離が近いポイントを取り出す
            if distance < nearest:
                point = target
                nearest = distance

        # パーティクルフィルタを更新
        if point is not None:
            x = point[0]
            y = point[1]
            self.pf.predict(x, y)

        return point

    def track_person(self, person_id):
        person = self.persons[person_id]
        x = person[0]
        y = person[1]

        self.pf = pf.ParticleFilter(x, y)

        track = [person]
        particles = [self.pf.particles]
        for keypoint_lst in self.keypoints_frame_lst:
            point = self._track(x, y, keypoint_lst)
            track.append(point)
            particles.append(self.pf.particles)

        return track, particles
