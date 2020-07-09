import numpy as np
import mahalanobis as mh


class ParticleFilter:
    def __init__(self, x, y, n_particle=300, sigma=[[50, 0], [0, 50]], noize_sigma=10):
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

    def _liklihood(self, x, y):
        mu = np.array([x, y])
        det = np.linalg.det(self.sigma)
        inv = np.linalg.inv(self.sigma)
        n = self.particles.ndim
        self.weights = np.exp(
            -np.diag((self.particles - mu) @ inv @ (self.particles - mu).T) / 2.0
        ) / (np.sqrt((2 * np.pi) ** n * det))

    def _resample(self):
        tmp_particles = self.particles.copy()
        w_cumsum = self.weights.cumsum()
        for i in range(self.n_particle):
            u = np.random.rand() * w_cumsum[-1]
            self.particles[i] = tmp_particles[(w_cumsum > u).argmax()]

    def _predict(self):
        x_noize = np.random.normal(0, np.sqrt(self.noize_sigma), self.n_particle)
        y_noize = np.random.normal(0, np.sqrt(self.noize_sigma), self.n_particle)
        v = np.stack([x_noize, y_noize], axis=1)
        self.particles = self.particles + v

    def predict(self, x, y):
        # 尤度関数(ガウス分布)からパーティクルの重みを計算
        self._liklihood(x, y)
        # パーティクルをリサンプリング
        self._resample()
        # 次の動きをランダムに予測する(パーティクルにノイズを乗せる)
        self._predict()

    def filtered_value(self):
        # パーティクルの加重平均から推定値を求める
        return np.average(self.particles.T, weights=self.weights, axis=1)

    def mahalanobis(self, point):
        return mh.calc(point, self.particles)
