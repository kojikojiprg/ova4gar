from keypoint import KeypointsList
from particle_filter import ParticleFilter
from heatmap import Heatmap
from functions import euclidean, cosine, normalize, softmax
import numpy as np
from enum import Enum, auto


class Person:
    def __init__(self, person_id, keypoints, max_age=5, vector_size=5):
        self.state = State.Reset
        self.id = person_id
        self.age = 0
        self.max_age = max_age

        self.keypoints_lst = KeypointsList()

        point = self._get_point(keypoints)
        self.pf = ParticleFilter(point)
        self.particles_lst = []
        self.mean_lst = []

        self.vector_size = vector_size
        self.vector_lst = []
        self.vector = np.array([0, 0])

        self.heatmap = Heatmap()
        self.vector_map = []
        self.move_hand_map = []

    def _get_point(self, keypoints):
        return keypoints.get_middle('Hip')

    def reset(self):
        if not self.is_deleted():
            self.state = State.Reset

    def is_reset(self):
        return self.state == State.Reset

    def is_deleted(self):
        return self.state == State.Deleted

    def probability(self, point, th):
        prob = self.pf.liklihood(point).sum()
        return prob if prob >= th else 0.0

    def update(self, keypoints):
        self.keypoints_lst.append(keypoints)

        self.pf.predict(self.vector)
        self.particles_lst.append(self.pf.particles.copy())

        if keypoints is not None:
            point = self._get_point(keypoints)
            x = self.pf.filter(point)
            self.mean_lst.append(x)
            self.age = 0
        else:
            x = self.pf.weighted_average()
            self.mean_lst.append(x)
            self.age += 1

        self.calc_vector()

        self.vector_map.append(self.heatmap.vector(self.vector, x))
        self.move_hand_map.append(self.heatmap.move_hand(keypoints))

        # ageがmax_ageを超えると削除
        if self.age > self.max_age:
            self.state = State.Deleted
        else:
            self.state = State.Updated

    def delete(self):
        self.keypoints_lst.append(None)
        self.particles_lst.append(None)
        self.mean_lst.append(None)
        self.vector_lst.append(None)
        self.vector_map.append(None)
        self.move_hand_map.append(None)

    def calc_vector(self):
        if self.is_deleted() or len(self.mean_lst) < self.vector_size:
            self.vector_lst.append(None)
            return

        # 差分を求める
        means = self.mean_lst[-self.vector_size:]
        diffs = []
        for i in range(self.vector_size - 1):
            now = means[i]
            nxt = means[i + 1]
            diffs.append(nxt - now + 0.00000001)

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
        self.vector = vec


class State(Enum):
    Reset = auto()
    Updated = auto()
    Deleted = auto()
