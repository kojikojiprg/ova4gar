import numpy as np
from scipy.linalg import solve_triangular


def calc(data):
    mu = np.average(data, axis=1)
    sigma = np.cov(data, rowvar=True, bias=True)

    L = np.linalg.cholesky(sigma)
    d = data - mu
    z = solve_triangular(
        L, d.T, lower=True, check_finite=False,
        overwrite_b=True)
    squared_maha = np.sum(z * z, axis=0)
    return squared_maha
