"""
pyron.utils
===========

This module contains various transformation, normalization, and activation functions.
"""

import math
from typing import List

import numpy as np
from numpy.typing import NDArray


def z_transform(X: NDArray, degree: int = 2) -> NDArray:
    """Z-transform a dataset `X`` into `degree` space.

    :param X: a dataset represented as a numpy.NDArray
    :param degree: an integer degree to transform to
    """

    if degree < 2:
        return X

    Z: NDArray = X.copy()
    d: int = len(X[0])

    B: List[int] = []
    for i in range(degree):
        B.append(math.comb(d + i, d - 1))

    l: NDArray = np.arange(np.sum(B))

    q: int = 0
    p: int = d
    g: int = d
    for i in range(1, degree):
        for j in range(q, q + p):
            for k in range(l[j], d):
                t: NDArray = Z[:, j] * X[:, k]
                Z = np.append(Z, t.reshape(-1, 1), 1)
                l[g] = k
                g = g + 1

        q = q + p
        p = B[i]

    return Z


def normalize_0_1(X: NDArray) -> NDArray:
    n, d = X.shape
    X_norm = X.astype("float64")

    for i in range(d):
        col_min = min(X_norm[:, i])
        col_max = max(X_norm[:, i])

        gap = col_max - col_min
        X_norm[:, i] = (X_norm[:, i] - col_min) / gap if gap else 0

    return X_norm


def normalize_neg1_pos1(X: NDArray) -> NDArray:
    n, d = X.shape
    X_norm = X.astype("float64")

    for i in range(d):
        col_min = min(X_norm[:, i])
        col_max = max(X_norm[:, i])
        col_mid = (col_max + col_min) / 2

        gap = (col_max - col_min) / 2
        X_norm[:, i] = (X_norm[:, i] - col_mid) / gap if gap else 0

    return X_norm


def tanh(x: NDArray) -> NDArray:
    """Vectorized hyperbolic tangent function

    :param x: an array type of real numbers

    :return: the numpy array where every element is tanh of the corresponding element in array x
    """
    return np.array(np.vectorize(np.tanh)(x))


def tanh_de(x: NDArray) -> NDArray:
    """Derivative of the hyperbolic tangent function

    :param x: an array type of real numbers

    :return: the numpy array where every element is tanh derivative of the corresponding element in array x
    """
    return x


def logis(x: NDArray) -> NDArray:
    """Vectorized logistic function

    :param x: an array type of real numbers

    :return: the numpy array where every element is logistic of the corresponding element in array x
    """
    return np.vectorize(lambda s: 1 / (1 + np.exp(-s)))(x)


def logis_de(x: NDArray) -> NDArray:
    """Derivative of the logistic function

    :param x: an array type of real numbers

    :return: the numpy array where every element is logistic derivative of the
             corresponding element in array x
    """
    return x


def iden(x: NDArray) -> NDArray:
    """Vectorized identity function

    :param x: an array type of real numbers

    :return: the numpy array where every element is the same as the corresponding element in array x
    """
    return np.vectorize(np.identity)(x)


def iden_de(x: NDArray) -> NDArray:
    """The derivative of the identity function

    :param x: an array type of real numbers

    :return: the numpy array of all zeros of the same shape of x.
    """
    return x


def relu(x: NDArray) -> NDArray:
    """Vectorized ReLU function

    :param x: an array type of real numbers

    :return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
    """
    return np.vectorize(np.maximum)(0, x)


def _relu_de_scaler(x: NDArray) -> NDArray:
    """The derivative of the ReLU function. Scaler version.

    :param x: a real number

    :return: 1, if x > 0; 0, otherwise.
    """
    return x


def relu_de(x: NDArray) -> NDArray:
    """The derivative of the ReLU function

    :param x: an array type of real numbers

    :return: the numpy array where every element is the _relu_de_scaler of the corresponding element in x.
    """
    return x
