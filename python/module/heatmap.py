import numpy as np


class Heatmap(list):
    def __init__(self, distribution, hip2ankle=50):
        super().__init__([])
        self._calc_args(distribution)
        self.hip2ankle = hip2ankle

    def _calc_args(self, distribution):
        self.xmax = np.nanmax(distribution)
        self.xmin = np.nanmin(distribution)
        half = (self.xmax - self.xmin) / 2
        self.inclination = 255 / half
        self.xmid = half + self.xmin

    def _colormap(self, x):
        if x <= self.xmid:
            r = 0
            g = self.inclination * (x - self.xmin)
            b = self.inclination * (self.xmid - x)
        else:
            r = self.inclination * (x - self.xmid)
            g = self.inclination * (self.xmax - x)
            b = 0
        return (int(r), int(g), int(b))


class Vector(Heatmap):
    def __init__(self):
        super().__init__([0.0, np.pi / 2.0])

    def calc(self, vec, mean_point):
        if vec is None or mean_point is None:
            return None

        angle = np.arccos(np.abs(vec[0]) / (np.linalg.norm(vec) + 0.00000001))

        start = mean_point
        start[1] += self.hip2ankle
        end = start + vec

        self.append((start, end, self._colormap(angle)))


class MoveHand(Heatmap):
    def __init__(self):
        super().__init__([0.0, np.pi])

    def calc(self, keypoints):
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
        self.append((point, self._colormap(angle)))


class Population(Heatmap):
    def __init__(self, homography, bins=(8, 8)):
        super().__init__([0.0, 3.0])
        self.homo = homography
        self.bins = bins
        size = list(homography.size)
        self.range = [[0, size[0]], [0, size[1]]]

    def calc(self, targets):
        # ターゲットにホモグラフィ変換する
        points = []
        for target in targets:
            target[1] += self.hip2ankle
            p = self.homo.transform_point(target)
            points.append(p)
        x, y = zip(*points)

        # 密度を計算
        H, xedges, yedges = np.histogram2d(x, y, self.bins, self.range)

        # 四角形の頂点とカラーマップを求める
        ds = []
        for i in range(self.bins[0]):
            for j in range(self.bins[1]):
                if H[i][j] > 0:
                    # 密度が0以上の場合のみ追加
                    p1 = (int(xedges[i]), int(yedges[j]))
                    p2 = (int(xedges[i + 1]), int(yedges[j + 1]))
                    ds.append((p1, p2, self._colormap(H[i][j])))

        self.append(ds)
