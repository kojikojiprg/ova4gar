import numpy as np


class Heatmap(list):
    def __init__(self, distribution, hip2ankle=50):
        super().__init__([])
        self._calc_args(distribution)

    def _calc_args(self, distribution):
        self.xmax = np.nanmax(distribution)
        self.xmin = np.nanmin(distribution)
        half = (self.xmax - self.xmin) / 2
        self.inclination = 255 / half
        self.xmid = half + self.xmin

    def colormap(self, x):
        if x <= self.xmid:
            r = 0
            g = self.inclination * (x - self.xmin)
            b = self.inclination * (self.xmid - x)
        else:
            r = self.inclination * (x - self.xmid)
            g = self.inclination * (self.xmax - x)
            b = 0
        return (int(r), int(g), int(b))


class Population(Heatmap):
    def __init__(self, homography, bins=(8, 8)):
        super().__init__([0.0, 4.0])
        self.homo = homography
        self.bins = bins
        size = list(homography.size)
        self.range = [[0, size[0]], [0, size[1]]]

    def calc(self, keypoints_lst):
        # ターゲットにホモグラフィ変換する
        points = []
        for keypoints in keypoints_lst:
            p = keypoints.get_middle('Ankle')
            p = self.homo.transform_point(p)
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
                    ds.append((p1, p2, self.colormap(H[i][j])))

        self.append(ds)
