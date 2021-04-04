import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing as pp


def mahalanobis(x, data):
    mu = np.average(data, axis=0)
    sigma = np.cov(data.T)

    inv = np.linalg.inv(sigma)
    return distance.mahalanobis(x, mu, inv)


def euclidean(a, b, axis=None):
    return np.linalg.norm(a - b, axis=axis)


def cosine(a, b):
    return distance.cosine(a, b)


def normalize(x):
    return pp.minmax_scale(x)


def standardize(x):
    return pp.scale(x)


def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def normalize_vector(vec):
    vec += 1e-10
    vec /= np.linalg.norm(vec)
    return vec


def rotation(vec, rad):
    R = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]])

    return np.dot(R, vec)
