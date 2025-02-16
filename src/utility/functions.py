import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing as pp


def mahalanobis(x, data):
    mu = np.average(data, axis=0)
    sigma = np.cov(data.T)

    inv = np.linalg.pinv(sigma)
    return distance.mahalanobis(x, mu, inv)


def gauss(x, a=1, mu=1, sigma=1):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


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
    vec = vec.astype(np.float64)
    vec /= np.linalg.norm(vec) + 1e-10
    return vec


def cos_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / (norm + 1e-10)


def rotation(vec, rad):
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    return np.dot(R, vec)


def moving_average(data, size):
    data_mean = np.convolve(data, np.ones(size) / size, mode="same")
    n_conv = size // 2

    # correction
    data_mean[0] *= size / n_conv
    for i in range(1, n_conv):
        data_mean[i] *= size / (i + n_conv)
        data_mean[-i] *= size / (i + n_conv - (size % 2))

    return data_mean
