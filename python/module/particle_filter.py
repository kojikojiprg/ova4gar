import numpy as np


class ParticleFilter:
    def __init__(
        self,
        point,
        n_particle=100,
        cov_gaus=np.eye(2) * 50,
        cov_predict=np.eye(2) * 10
    ):
        self.n_particle = n_particle    # number of particle
        self.cov_gaus = cov_gaus        # var of gaussian
        self.cov_predict = cov_predict  # var of noize

        # init particles
        self.particles = np.random.multivariate_normal(
            point, self.cov_predict, self.n_particle)
        self.weights = np.ones(self.n_particle) / self.n_particle

    def _resample(self):
        tmp_particles = self.particles.copy()
        w_cumsum = self.weights.cumsum()
        for i in range(self.n_particle):
            u = np.random.rand() * w_cumsum[-1]
            self.particles[i] = tmp_particles[(w_cumsum > u).argmax()]

    def _predict(self):
        self.particles += np.random.multivariate_normal(
            np.zeros(2), self.cov_predict, self.n_particle)
        self.weights = np.ones(self.n_particle) / self.n_particle

    def predict(self):
        # パーティクルをリサンプリング
        self._resample()

        # 次の動きをランダムに予測する(パーティクルにノイズを乗せる)
        self._predict()

    def liklihood(self, point):
        mu = np.array(point)
        det = np.linalg.det(self.cov_gaus)
        inv = np.linalg.inv(self.cov_gaus)
        n = self.particles.ndim
        return np.exp(
            -np.diag((self.particles - mu) @ inv @ (self.particles - mu).T) / 2.0
        ) / (np.sqrt((2 * np.pi) ** n * det))

    def filter(self, point):
        # 尤度関数(ガウス分布)からパーティクルの重みを計算
        self.weights = self.liklihood(point)

        # パーティクルの加重平均から推定値を求める
        return np.average(self.particles.T, weights=self.weights, axis=1)
