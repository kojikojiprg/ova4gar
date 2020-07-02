import numpy as np
import particle_filter as pf
import mahalanobis as mh


class Tracker:
    def __init__(self, keypoints_frame_lst, init_gate_size):
        self.keypoints_frame_lst = keypoints_frame_lst

        keypoints_lst = keypoints_frame_lst[0]
        self.persons = self._gate_keypoint_lst(keypoints_lst, init_gate_size)

    def _obtain_point(self, keypoints):
        R = np.array(keypoints.get('RAnkle'))
        L = np.array(keypoints.get('LAnkle'))
        if R[2] == 0.0:
            point = L
        elif L[2] == 0.0:
            point = R
        else:
            point = (R + L) / 2
        return point[:2].astype(int)

    def _is_gate(self, point, size):
        x = point[0]
        y = point[1]
        size = np.array(size)
        return (
            (size[0, 0] <= x and x <= size[1, 0]) and
            (size[0, 1] <= y and y <= size[1, 1]))

    def _gate_keypoint_lst(self, keypoints_lst, size):
        gate_through = []
        for keypoints in keypoints_lst:
            point = self._obtain_point(keypoints)
            if self._is_gate(point, size):
                gate_through.append(point)
        return gate_through

    def track_person(self, person_id, gate_size=50):
        person = self.persons[person_id]
        x = person[0]
        y = person[1]

        self.pf = pf.ParticleFilter(x, y)

        track = [person]
        for keypoint_lst in self.keypoints_frame_lst:
            # 近くのキーポイントのみ抽出
            x1 = x - int(gate_size / 2)
            x2 = x + int(gate_size / 2)
            y1 = y - int(gate_size / 2)
            y2 = y + int(gate_size / 2)
            targets = self._gate_keypoint_lst(keypoint_lst, ((x1, y1), (x2, y2)))

            nearest = np.inf
            point = None
            for target in targets:
                # マハラノビス距離を求める
                distance = mh.calc(target, self.pf.particles)

                if distance < nearest:
                    point = target
                    nearest = distance

            track.append(point)

            # パーティクルフィルタを更新
            if point is not None:
                x = point[0]
                y = point[1]
                self.pf.predict(x, y)

        return track
