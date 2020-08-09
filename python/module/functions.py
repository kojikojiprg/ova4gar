import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing as pp


def mahalanobis(x, data):
    mu = np.average(data, axis=0)
    sigma = np.cov(data.T)

    inv = np.linalg.inv(sigma)
    return distance.mahalanobis(x, mu, inv)


def euclidean(a, b):
    return np.linalg.norm(b - a)


def cosine(a, b):
    return distance.cosine(a, b)


def normalize(x):
    return pp.minmax_scale(x)


def standardize(x):
    return pp.scale(x)
