from common.keypoint import KeypointsList
from common.functions import euclidean, cosine, normalize, softmax
from tracker.particle_filter import ParticleFilter
import numpy as np
from enum import Enum, auto


class Person:
    def __init__(self, person_id, keypoints, max_age=10, vector_size=10):
        self.state = State.Reset
        self.id = person_id
        self.age = 0
        self.max_age = max_age

        self.keypoints_lst = KeypointsList()

        point = self._get_point(keypoints)
        self.pf = ParticleFilter(point)
        self.particles_lst = []
        self.average_lst = []

        self.vector_size = vector_size
        self.vector = np.array([0, 0])

    def _get_point(self, keypoints):
        return keypoints.get_middle('Hip')

    def reset(self):
        if not self.is_deleted():
            self.state = State.Reset

    def is_reset(self):
        return self.state == State.Reset

    def is_deleted(self):
        return self.state == State.Deleted

    def probability(self, point):
        prob = self.pf.liklihood(point)
        prob = prob.sum()
        return prob

    def update(self, keypoints):
        self.keypoints_lst.append(keypoints)

        self.pf.predict(self.vector)
        self.particles_lst.append(self.pf.particles.copy())

        if keypoints is not None:
            point = self._get_point(keypoints)
            x = self.pf.filter(point)
            self.average_lst.append(x)
            self.age = 0
        else:
            x = self.pf.weighted_average()
            self.average_lst.append(x)
            self.age += 1

        self.calc_vector()

        # ageがmax_ageを超えると削除
        if self.age > self.max_age:
            self.state = State.Deleted
            self.delete()
        else:
            self.state = State.Updated

    def delete(self):
        for i in range(1, self.max_age):
            self.keypoints_lst[-i] = None
            self.particles_lst[-i] = None
            self.average_lst[-i] = None

    def update_deleted(self):
        self.keypoints_lst.append(None)
        self.particles_lst.append(None)
        self.average_lst.append(None)

    def calc_vector(self):
        if self.is_deleted() or len(self.average_lst) < self.vector_size:
            return

        # 差分を求める
        average = self.average_lst[-self.vector_size:]
        diffs = []
        for i in range(self.vector_size - 1):
            now = average[i]
            nxt = average[i + 1]
            diffs.append(nxt - now + 1e-10)

        # 類似度を計算
        euclidieans = []
        cosines = []
        for a, b in zip(diffs[:-1], diffs[1:]):
            euclidieans.append(euclidean(a, b))
            cosines.append(cosine(a, b))

        # ユークリッド距離を正規化
        euclidieans = normalize(euclidieans)

        # ユークリッド距離 * コサイン類似度の逆数を重みとする(0 ~ 1)
        weights = 1 / (np.array(euclidieans) * np.array(cosines) + 1e-10)
        # 重みの合計を1にする
        weights = softmax(weights)

        # ベクトルを求める
        vec = np.average(diffs[1:], weights=weights, axis=0)
        # 見つからない状態が続くごとにベクトルを小さくする
        vec = vec / (self.age + 1)
        vec = vec.astype(int)

        self.vector = vec


class State(Enum):
    Reset = auto()
    Updated = auto()
    Deleted = auto()