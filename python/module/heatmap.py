import numpy as np


class Heatmap:
    def __init__(self):
        self.vector_args = self._calc_args([0.0, np.pi / 2])
        self.move_hand_args = self._calc_args([0.0, np.pi])
        self.hip2ankle = 50

    def _calc_args(self, distribution):
        xmax = np.nanmax(distribution)
        xmin = np.nanmin(distribution)
        half = (xmax - xmin) / 2
        inclination = 255 / half
        xmid = half + xmin
        return [xmin, xmax, xmid, inclination]

    def _colormap(self, x, args):
        xmin = args[0]
        xmax = args[1]
        xmid = args[2]
        inclination = args[3]
        if x <= xmid:
            r = 0
            g = inclination * (x - xmin)
            b = inclination * (xmid - x)
        else:
            r = inclination * (x - xmid)
            g = inclination * (xmax - x)
            b = 0
        return (int(r), int(g), int(b))

    def vector(self, vec, mean_point):
        if vec is None or mean_point is None:
            return None

        angle = np.arccos(np.abs(vec[0]) / (np.linalg.norm(vec) + 0.00000001))

        start = mean_point
        start[1] += self.hip2ankle
        end = start + vec

        return start, end, self._colormap(angle, self.vector_args)

    def move_hand(self, keypoints):
        if keypoints is None:
            return None

        mid_shoulder = keypoints.get_middle('Shoulder')
        mid_hip = keypoints.get_middle('Hip')

        # 体軸ベクトルとノルム
        axis = mid_shoulder - mid_hip
        norm_axis = np.linalg.norm(axis, ord=2)

        ankle = 0.
        for side in ('R', 'L'):
            elbow = keypoints.get(side + 'Elbow', ignore_confidence=True)
            wrist = keypoints.get(side + 'Wrist', ignore_confidence=True)

            # 前肢ベクトルとノルム
            vec = wrist - elbow
            norm = np.linalg.norm(vec, ord=2)

            # 体軸と前肢の角度(左右の大きい方を選択する)
            angle = max(
                ankle,
                np.arccos(np.dot(axis, vec) / (norm_axis * norm + 0.00000001)))

        point = mid_hip + self.hip2ankle
        return point, self._colormap(angle, self.move_hand_args)
