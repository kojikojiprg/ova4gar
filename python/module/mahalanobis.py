import numpy as np
from scipy.spatial.distance import mahalanobis


def calc(x, data):
    mu = np.average(data, axis=0)
    sigma = np.cov(data.T)

    inv = np.linalg.inv(sigma)
    print(sigma)
    return mahalanobis(x, mu, inv)
