import numpy as np
import particle_filter as pf
import mahalanobis as mh


class Tracker:
    def __init__(self, keypoints_frame_lst):
        self.keypoints_frame_lst = keypoints_frame_lst

        keypoints_lst = keypoints_frame_lst[0]
        self.persons = self._obtain_targets(keypoints_lst)

    def _obtain_point(self, keypoints):
        R = np.array(keypoints.get('RAnkle'))
        L = np.array(keypoints.get('LAnkle'))
        if R[2] < 0.05:
            point = L
        elif L[2] < 0.05:
            point = R
        else:
            point = (R + L) / 2
        return point[:2].astype(int)

    def _obtain_targets(self, keypoints_lst):
        targets = []
        for keypoints in keypoints_lst:
            point = self._obtain_point(keypoints)
            targets.append(point)
        return targets

    def _track(self, x, y, keypoint_lst):
        targets = self._obtain_targets(keypoint_lst)

        nearest = np.inf
        point = None
        for target in targets:
            # マハラノビス距離を求める
            distance = mh.calc(target, self.pf.particles)

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
