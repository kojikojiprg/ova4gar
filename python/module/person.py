from keypoint import KeypointsList
from particle_filter import ParticleFilter
from functions import euclidean, cosine, normalize, softmax
import numpy as np
from enum import Enum, auto


class Person:
    def __init__(self, person_id, keypoints, vector_size=10, frame_num=0):
        self.state = State.Reset
        self.id = person_id

        self.keypoints_lst = KeypointsList()

        point = self._get_point(keypoints)
        self.pf = ParticleFilter(point)
        self.particles_lst = []

        self.vector_size = vector_size
        self.vector_lst = []

        for _ in range(frame_num):
            self.keypoints_lst.append(None)
            self.particles_lst.append(None)
            self.vector_lst.append(None)

    def _get_point(self, keypoints):
        return keypoints.get_middle('Hip')

    def reset(self):
        self.state = State.Reset

    def is_updated(self):
        return self.state == State.Updated

    def probability(self, point, th):
        weights = self.pf.liklihood(point)
        if weights.sum() >= th:
            return weights.sum()
        else:
            return 0.0

    def update(self, keypoints):
        self.keypoints_lst.append(keypoints)

        self.pf.predict()
        self.particles_lst.append(self.pf.particles.copy())

        if keypoints is not None:
            point = self._get_point(keypoints)
            self.pf.filter(point)

        # Noneの処理を考える
        #self.vector()
        self.state = State.Updated

    def vector(self):
        if len(self.keypoints_lst) < self.vector_size:
            self.vector_lst.append(None)
            return

        # 差分を求める
        kp_reversed = self.keypoints_lst[::-1]
        diffs = []
        for i in range(self.vector_size):
            now = self._get_point(kp_reversed[i])
            nxt = self._get_point(kp_reversed[i - 1])
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
        vec = np.average(diffs[1:], weights=weights, axis=0)
        vec = vec.astype(int)

        self.vector_lst.append(tuple(vec))


class State(Enum):
    Reset = auto()
    Updated = auto()
    Deleted = auto()
