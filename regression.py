"""
pyron.regression
================

This module contains two classes, one implementing a linear regression and the other a logistic regression,
the former featuring closed-form and gradient descent methods of fitting, and the latter featuring
batch gradient descent and stochastic gradient descent methods of fitting.
"""

import math
import sys

from utils import z_transform

import numpy as np


class LinearRegression:
    def __init__(self) -> LinearRegression:
        """Initialize `LinearRegression`"""
        self.w = None
        self.degree = 1

    def fit(self, X, y, CF=True, lam=0, eta=0.01, epochs=1000, degree=1):
        self.degree = degree
        X = z_transform(X, degree=self.degree)

        if CF:
            self._fit_cf(X, y, lam)
        else:
            self._fit_gd(X, y, lam, eta, epochs)

    def _fit_cf(self, X, y, lam=0):
        """Fit with closed form method"""
        X = np.insert(X, 0, 1, axis=1)
        X_t = np.transpose(X)
        self.w = np.linalg.pinv(
            X_t @ X + (lam * np.identity(len(X[0])))) @ X_t @ y

    def _fit_gd(self, X, y, lam=0, eta=0.01, epochs=1000):
        """Fit with gradient descent method"""
        X = np.insert(X, 0, 1, axis=1)
        X_t = np.transpose(X)
        n, d = X.shape
        X_t = np.transpose(X)
        I = np.identity(d, dtype=int)
        k = 2 * eta / n

        self.w = np.ones((d, 1), dtype=int)

        a = (I - k * (X_t + (lam * I))) @ X
        b = k * X_t @ y

        for i in range(epochs):
            self.w = a @ self.w + b

    def predict(self, X):
        X = z_transform(X, degree=self.degree)
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.w

    def error(self, X, y):
        X = z_transform(X, degree=self.degree)
        X = np.insert(X, 0, 1, axis=1)
        n, d = X.shape
        return np.sum(np.power(X @ self.w - y, 2)) / n


class LogisticRegression:
    def __init__(self) -> LogisticRegression:
        """Initialize `LogisticRegression`"""
        self.w: NDArray = None
        self.degree: int = 1

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        lam: int = 0,
        eta: float = 0.01,
        iterations: int = 1000,
        SGD: bool = False,
        mini_batch_size: int = 1,
        degree: int = 1,
    ):
        self.degree = degree
        X = z_transform(X, degree=self.degree)
        n, d = X.shape
        X = np.insert(X, 0, 1, axis=1)
        y = np.array(y).reshape(-1, 1)
        self.w = np.zeros((d + 1, 1))

        if SGD:
            mini_batch_size = (
                n if mini_batch_size > n or mini_batch_size < 1 else mini_batch_size
            )
            self._fit_sgd(X, y, lam, eta, iterations, mini_batch_size)
        else:
            self._fit_bgd(X, y, lam, eta, iterations)

    def _fit_bgd(self, X: NDArray, y: NDArray, lam: int, eta: float, iterations: int):
        """Fit with batch gradient descent"""
        n: int = len(y)
        step: int = eta / len(y)
        for i in range(iterations):
            s: NDArray = y * (X @ self.w)
            self.w = (1 - ((2 * lam * eta) / n)) * self.w + step * (
                X.T @ (y * LogisticRegression._vsigmoid(-s))
            )

    def _fit_sgd(self, X, y, lam, eta, iterations, mini_batch_size):
        """Fit with stochastic gradient descent"""
        n: int = len(y)
        n_blocks: int = math.ceil(n / mini_batch_size)
        for i in range(iterations):
            start: int = (i % n_blocks) * mini_batch_size
            end: int = min(start + mini_batch_size, n)
            n_p: int = end - start
            step: int = eta / n_p

            X_p: NDArray = X[start:end, :]
            y_p: NDArray = y[start:end, :]

            s: NDArray = y_p * (X_p @ self.w)
            self.w = (1 - ((2 * lam * eta) / n_p)) * self.w + step * (
                y_p * LogisticRegression.v_sigmoid(-s).T @ X_p
            ).T

    def predict(self, X: NDArray) -> NDArray:
        X = z_transform(X, degree=self.degree)
        X = np.insert(X, 0, 1, axis=1)
        return LogisticRegression.v_sigmoid(X @ self.w)

    def error(self, X, y) -> int:
        misclassified: int = 0
        signals: NDArray = self.predict(X)
        for i in range(len(y)):
            signals[i] = 1 if signals[i] > 0.5 else -1
            if signals[i] != y[i]:
                misclassified += 1

        return misclassified

    @staticmethod
    def v_sigmoid(s) -> NDArray:
        """Vectorized sigmoid activation function"""
        return np.vectorize(LogisticRegression.sigmoid)(s)

    @staticmethod
    def sigmoid(s) -> float:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-s))
