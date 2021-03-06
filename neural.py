"""
pyron.network
=============

This module contains two classes necessary for composing a rudimentary neural network:
NeuralLayer, which represents a single layer of a network, and NeuralNetwork, the collection of
layers, state, and functions relevant to the network.
"""

import math
from typing import Any, List

import utils

import numpy as np
from numpy.typing import NDArray


class NeuralLayer:
    def __init__(self, d: int = 1, act: str = "tanh"):
        """Initialize `NeuralLayer`

        :param d: the number of NON-bias nodes in the layer
        :param act: the activation function. It will not be useful/used, regardlessly, at the input layer.

        Available activation functions include:
        - 'tanh': the tanh function
        - 'logis': the logistic function
        - 'iden': the identity function
        - 'relu': the ReLU function
        """
        self.d: int = d

        self.act: Any = eval("utils." + act)
        self.act_de: Any = eval("utils." + act + "_de")

        self.S: NDArray = None
        self.X: NDArray = None
        self.Delta: NDArray = None
        self.G: NDArray = None
        self.W: NDArray = None


class NeuralNetwork:
    def __init__(self):
        """Initialize `NeuralNetwork`"""
        self.layers: List[NeuralLayer] = []
        self.L: int = -1

    def add_layer(self, d: int = 1, act: str = "tanh"):
        """The newly added layer is always added AFTER all existing layers.
        The firstly added layer is the input layer.
        The most recently added layer is the output layer.

        :param d: number of nodes excluding bias node
        :param act: activation function to be supplied by the user (default: tanh)

        Available activation functions include:
        - 'tanh': the tanh function
        - 'logis': the logistic function
        - 'iden': the identity function
        - 'relu': the ReLU function
        """
        self.layers = np.append(self.layers, NeuralLayer(d=d, act=act))
        self.L = self.L + 1

    def _init_weights(self):
        """Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)],
        where d is the number of nonbias node of the layer
        """
        for i, layer in enumerate(self.layers):
            n: int = self.layers[i - 1].d + 1
            d: int = layer.d
            a: NDArray = np.random.uniform(
                -1 / np.sqrt(layer.d), 1 / np.sqrt(layer.d), n * d
            )
            layer.W = a.reshape((n, d))

    def fit(
        self,
        X: List[List[int]],
        Y: List[List[int]],
        eta: float = 0.01,
        iterations: int = 1000,
        SGD: bool = True,
        mini_batch_size: int = 1,
    ):
        """Find the fitting weight matrices for every hidden layer and the output layer
        and save them in the layers.

        :param X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
        :param Y: n x k vector of labels, where k >= 1 is the number of classes
        :param eta: the learning rate used in gradient descent
        :param iterations: the maximum iterations used in gradient descent
        :param SGD: whether or not to use SGD (default True)
        :param mini_batch_size: the size of each mini batch size, if SGD is True.
        """
        self._init_weights()
        n, _ = X.shape
        mini_batch_size = (
            n if mini_batch_size > n or mini_batch_size < 1 else mini_batch_size
        )
        n_blocks: int = math.ceil(n / mini_batch_size)

        for i in range(iterations):
            start: int = (i % n_blocks) * mini_batch_size
            end: int = min(start + mini_batch_size, n)
            n_p: int = end - start
            step: int = eta / n_p

            X_p: NDArray = np.insert(X[start:end, :], 0, 1, axis=1)
            Y_p: NDArray = Y[start:end, :]

            self.layers[0].X = X_p
            for l in range(1, self.L + 1):
                self.layers[l].S = self.layers[l - 1].X @ self.layers[l].W
                self.layers[l].X = np.insert(
                    self.layers[l].act(self.layers[l].S), 0, 1, axis=1
                )

            self.layers[self.L].Delta = (
                2
                * (self.layers[self.L].X[:, 1:] - Y_p)
                * (self.layers[self.L].act_de(self.layers[self.L].S))
            )

            self.layers[self.L].G = np.einsum(
                "ij,ik -> jk", self.layers[self.L -
                                           1].X, self.layers[self.L].Delta
            ) * (1 / n_p)

            for l in range(self.L - 1, 0, -1):
                self.layers[l].Delta = self.layers[l].act_de(self.layers[l].S) * (
                    self.layers[l +
                                1].Delta @ np.transpose(self.layers[l + 1].W)[:, 1:]
                )

                self.layers[l].G = np.einsum(
                    "ij,ik -> jk", self.layers[l - 1].X, self.layers[l].Delta
                ) * (1 / n_p)

            for l in range(1, self.L):
                self.layers[l].W = self.layers[l].W - eta * self.layers[l].G

    def predict(self, X: NDArray) -> NDArray:
        """
        :param X: n x d matrix, the sample batch, excluding the bias feature 1 column.

        :return: n x 1 matrix, n is the number of samples, every row is the predicted class ID.
        """
        X = np.insert(X, 0, 1, axis=1)
        self.layers[0].X = X
        for l in range(1, self.L + 1):
            self.layers[l].S = self.layers[l - 1].X @ self.layers[l].W
            self.layers[l].X = np.insert(
                self.layers[l].act(self.layers[l].S), 0, 1, axis=1
            )

        output: NDArray = np.zeros((len(X), 1))
        output = np.array([np.argmax(self.layers[self.L].X[i])
                          for i in range(len(X))])

        return output

    def error(self, X: NDArray, Y: NDArray) -> float:
        """
        :param X: n x d matrix, the sample batch, excluding the bias feature 1 column.
                  n is the number of samples.
                  d is the number of (non-bias) features of each sample.
        :param Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample,
                  where only one entry is 1 and the rest are all 0.
                  Y[i,j]=1 indicates the ith sample belongs to class j.
                  k is the number of classes.

        :return: the percentage of misclassfied samples
        """
        signals: NDArray = self.predict(X)

        err: NDArray = np.zeros((len(Y), 1))
        err = np.array([np.argmax(Y[i]) for i in range(len(Y))])

        return np.sum(signals != err) / len(Y)
