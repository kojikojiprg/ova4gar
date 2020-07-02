import numpy as np


class ParticleFilter:
    def __init__(self, x, y, n_particle=500, sigma=[[5, 0], [0, 5]], noize_sigma=10):
        self.n_particle = n_particle    # number of particle
        self.sigma = np.array(sigma)    # var of gaussian
        self.noize_sigma = noize_sigma  # var of noize

        # init particles
        self.particles = self._init_particles(x, y)
        self.weights = np.zeros((2, n_particle))

    def _init_particles(self, x, y):
        particles_x = np.random.normal(x, np.sqrt(self.noize_sigma), self.n_particle)
        particles_y = np.random.normal(y, np.sqrt(self.noize_sigma), self.n_particle)
        return np.stack([particles_x, particles_y], axis=1)

    def _add_noize(self):
        x = np.random.normal(0, np.sqrt(self.noize_sigma), self.n_particle)
        y = np.random.normal(0, np.sqrt(self.noize_sigma), self.n_particle)
        v = np.stack([x, y], axis=1)
        self.particles = self.particles + v

    def _liklihood(self, x, y):
        mu = np.array([x, y])
        det = np.linalg.det(self.sigma)
        inv = np.linalg.inv(self.sigma)
        n = self.particles.ndim
        self.weights = np.exp(
            -np.diag((self.particles - mu) @ inv @ (self.particles - mu).T) / 2.0
        ) / (np.sqrt((2 * np.pi) ** n * det))

    def _filtered_value(self):
        # パーティクルの加重平均
        return np.average(self.particles.T, weights=self.weights, axis=1)

    def _resample(self):
        idx = np.asanyarray(range(self.n_particle))
        u0 = np.random.uniform(0, 1 / self.n_particle)
        u = [1 / self.n_particle * i + u0 for i in range(self.n_particle)]
        w_cumsum = np.cumsum(self.weights)
        k = np.asanyarray([self._f_inv(w_cumsum, idx, val) for val in u])
        self.particles = self.particles[k]

    def _f_inv(self, w_cumsum, idx, u):
        if not np.any(w_cumsum < u):
            return 0

        k = np.max(idx[w_cumsum < u])
        return k

    def predict(self, x, y):
        # 全てのパーティクルにノイズを乗せる
        self._add_noize()
        # 尤度関数(ガウス分布)からパーティクルの重みを計算
        self._liklihood(x, y)
        # 推定値を取得
        v = self._filtered_value()
        # パーティクルをリサンプリング
        self._resample()
        self._add_noize()

        return v
