from keypoint import KeypointsList
from particle_filter import ParticleFilter
from functions import euclidean, cosine, normalize, softmax
import numpy as np


class Person:
    def __init__(self, person_id, keypoints, vector_size=10):
        self.id = person_id
        self.keypoints_lst = KeypointsList()
        self.keypoints_lst.append(keypoints)

        point = keypoints.get_middle('Hip')
        self.pf = ParticleFilter(point[0], point[1])
        self.particles_lst = [self.pf.particles]

        self.vector_size = vector_size
        self.vector_lst = [None]

    def update(self, point, keypoints):
        if point is not None:
            self.pf.predict(point[0], point[1])

        self.particles_lst.append(self.pf.particles)
        self.keypoints_lst.append(keypoints)
        self.vector()

    def vector(self):
        if len(self.keypoints_lst) < self.vector_size:
            self.vector_lst.append(None)
            return

        # 差分を求める
        kp_reversed = self.keypoints_lst[::-1]
        diffs = []
        for i in range(self.vector_size):
            now = kp_reversed[i].get_middle('Hip')
            nxt = kp_reversed[i - 1].get_middle('Hip')
            diffs.append(nxt - now + 0.00000001)
        diffs = np.array(diffs[::-1])

        # 類似度を計算
        euclidieans = []
        cosines = []
        for a, b in zip(diffs[:-1], diffs[1:]):
            euclidieans.append(euclidean(a, b))
            cosines.append(cosine(a, b))

        # ユークリッド距離を正規化
        euclidieans = normalize(euclidieans)

        # ユークリッド距離 * コサイン類似度の逆数を重みとする(0 ~ 1)
        weights = 1 / (np.array(euclidieans) * np.array(cosines) + 0.00000001)
        # 重みの合計を1にする
        weights = softmax(weights)

        # ベクトルを求める
        vec = 0.0
        for diff, weight in zip(diffs[1:], weights):
            vec += (diff * weight).astype(int)
        # 単位ベクトルに変換
        vec = vec.astype(int)

        self.vector_lst.append(tuple(vec))
