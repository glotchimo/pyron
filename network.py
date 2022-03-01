"""
pyron.network
=============

This module contains two classes necessary for composing a rudimentary neural network:
NeuralLayer, which represents a single layer of a network, and NeuralNetwork, the collection of
layers, state, and functions relevant to the network.
"""

import math
from typing import Any, List

import math_util as mu
import nn_layer

import numpy as np
from numpy.typing import NDArray


class NeuralLayer:
    def __init__(self, d: int = 1, act: str = "tanh") -> NeuralLayer:
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

        self.act: Any = eval("mu.MyMath." + act)
        self.act_de: Any = eval("mu.MyMath." + act + "_de")

        self.S: NDArray = None
        self.X: NDArray = None
        self.Delta = None
        self.G = None
        self.W = None


class NeuralNetwork:
    def __init__(self) -> NeuralNetwork:
        """Initialize `NeuralNetwork`"""
        self.layers: List[nn_layer.NeuralLayer] = []
        self.L: int = -1

    def add_layer(self, d: int = 1, act: str = "tanh"):
        """The newly added layer is always added AFTER all existing layers.
        The firstly added layer is the input layer.
        The most recently added layer is the output layer.

        :param d: number of nodes exluding bias node
        :param act: activation function to be supplied by the user (default: tanh)

        Available activation functions include:
        - 'tanh': the tanh function
        - 'logis': the logistic function
        - 'iden': the identity function
        - 'relu': the ReLU function
        """

    def _init_weights(self):
        """Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)],
        where d is the number of nonbias node of the layer
        """

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

    def predict(self, X: NDArray) -> NDArray:
        """
        :param X: n x d matrix, the sample batch, excluding the bias feature 1 column.

        :return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
        """
        X: NDArray = np.reshape(X, -1)
        for i in range(1, self.L + 1):
            self.layers[i - 1]

    def error(self, X: NDArray, Y: NDArray):
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
