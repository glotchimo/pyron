import sys

from neural import NeuralLayer, NeuralNetwork
from utils import (
    z_transform,
    tanh,
    tanh_de,
    logis,
    logis_de,
    iden,
    iden_de,
    relu,
    relu_de,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

verbose = True


def main():
    passed_math_util = test_math_util()
    assert passed_math_util
    passed_add_layer = test_add_layer()
    assert passed_add_layer
    passed_init_weights = test_init_weights()
    assert passed_init_weights
    passed_seeded_weights = test_seeded_weights()
    assert passed_seeded_weights

    (X_train, y_train, X_test, y_test) = loadData()


def test_seeded_weights():
    d = 2
    k = 2
    passed = True

    # build the network
    nuts = NeuralNetwork()

    nuts.add_layer(nodes=d)  # input layer - 0
    nuts.add_layer(nodes=5, act="relu")  # hidden layer - 1
    nuts.add_layer(nodes=k, act="logis")  # output layer

    nuts._init_weights()

    seed_weights = load_weights()
    for layer, seed_weight in zip(nuts.layers[1:], seed_weights):
        seed_weight = np.array(seed_weight)
        weight = np.array(layer.W)

        if (weight != seed_weight).any():
            if verbose:
                print(
                    f"check_seeded_weights:\nExpected:\n{seed_weight}\nFound:\n{weight}"
                )
            passed = False

    return passed


def test_init_weights():
    d = 10
    k = 8
    passed = True

    # build the network
    nuts = NeuralNetwork()

    nuts.add_layer(nodes=d)  # input layer - 0
    nuts.add_layer(nodes=5, act="relu")  # hidden layer - 1
    nuts.add_layer(nodes=k, act="logis")  # output layer

    nuts._init_weights()

    shapes = [(11, 5), (6, 8)]
    for layer, dim in zip(nuts.layers[1:], shapes):
        if layer.W.shape != dim:
            if verbose:
                print(
                    f"test_init_weights: Invalid dimensions of the instantiated weights. Expected {dim}, found {layer.W.shape}"
                )
            passed = False

    return passed


def test_add_layer():
    # build the network
    nuts = NeuralNetwork()

    assert nuts.L == -1, f"After initialization, L = -1. Found L = {nuts.L}"

    passed = True

    nuts.add_layer(nodes=5, act="logis")
    if nuts.L != 0:
        if verbose:
            print(
                f"test_add_layer: After adding a layer, L = 0. Found L = {nuts.L}")
        passed = False
    if len(nuts.layers) != 1:
        if verbose:
            print(f"test_add_layer: Failed to add layer to layers.")
        passed = False

    return passed


def test_math_util():
    passed = [
        test_tanh(),
        test_tanh_de(),
        test_logis(),
        test_logis_de(),
        test_iden(),
        test_iden_de(),
        test_relu(),
        test_relu_de(),
    ]
    if all(passed):
        return True
    else:
        print(
            f"test_math_util passed methods:\n\
tanh: {passed[0]}, tanh_de: {passed[1]}\n\
logis: {passed[2]}, logis_de: {passed[3]}\n\
iden: {passed[4]}, iden_de: {passed[5]}\n\
relu: {passed[6]}, relu_de: {passed[7]}"
        )

        return False


def test_tanh():
    x = [0.0, 1.0, 2.0, -1.0]
    y = [0.0, 0.761594156, 0.9640275801, -0.761594156]
    y_hat = tanh(x)
    return _test_math_util(x, y, y_hat, "tanh")


def test_tanh_de():
    x = [0.0, 1.0, 2.0, -1.0]
    y = [1, 0.419974, 0.0706508, 0.419974]
    y_hat = tanh_de(x)
    return _test_math_util(x, y, y_hat, "tanh_de")


def test_logis():
    x = [0.0, 1.0, 2.0, -1.0]
    y = [0.5, 0.7310586, 0.8807971, 0.2689414]
    y_hat = logis(x)
    return _test_math_util(x, y, y_hat, "logis")


def test_logis_de():
    x = [0.0, 1.0, 2.0, -1.0]
    y = [0.25, 0.196612, 0.104994, 0.196612]
    y_hat = logis_de(x)
    return _test_math_util(x, y, y_hat, "logis_de")


def test_iden():
    x = [0.0, 1.0, 2.0, -1.0]
    y = [0.0, 1.0, 2.0, -1.0]
    y_hat = iden(x)
    return _test_math_util(x, y, y_hat, "iden")


def test_iden_de():
    x = [0.0, 1.0, 2.0, -1.0]
    y = [1, 1, 1, 1]
    y_hat = iden_de(x)
    return _test_math_util(x, y, y_hat, "iden_de")


def test_relu():
    x = [0.0, 1.0, 2.0, -1.0]
    y = [0.0, 1.0, 2.0, 0.0]
    y_hat = relu(x)
    return _test_math_util(x, y, y_hat, "relu")


def test_relu_de():
    x = [0.0, 1.0, 2.0, -1.0]
    y = [0, 1, 1, 0]
    y_hat = relu_de(x)
    return _test_math_util(x, y, y_hat, "relu_de")


def _test_math_util(x, y, y_hat, name):

    passed = True

    if not _is_numpy_array(y_hat):
        if verbose:
            print(
                f"Incorrect return type. Expected type {np.ndarray}, but found {type(y_hat)}"
            )
        passed = False

    for y, y_hat, x in zip(y, y_hat, x):
        if not inThreshold(y, y_hat, 0.00001):
            if verbose:
                print(
                    f"Incorrect {name} value, expected {y}, but found {y_hat} for x = {x}"
                )
            passed = False
    return passed


def _is_numpy_array(x):
    return isinstance(x, np.ndarray)


def inThreshold(x1, x2, threshold=1.1):
    return abs(x1 - x2) < threshold


def saveAllZ(Z_list, file="output.npz"):
    np.savez(file, *Z_list)


def clearZ(file="output.npz"):
    f = open(file, "w")
    f.close()


def loadData(data_set="ionoshpere"):
    k = 10
    d = 784

    # Reads the files into pandas dataframes from the respective .csv files.
    path = "code_nn/MNIST"
    df_X_train = pd.read_csv(f"{path}/X_train.csv", header=None)
    df_y_train = pd.read_csv(f"{path}/y_train.csv", header=None)
    df_X_test = pd.read_csv(f"{path}/X_test.csv", header=None)
    df_y_test = pd.read_csv(f"{path}/y_test.csv", header=None)

    # save in numpy arrays
    X_train_raw = df_X_train.to_numpy()
    y_train_raw = df_y_train.to_numpy()
    X_test_raw = df_X_test.to_numpy()
    y_test_raw = df_y_test.to_numpy()

    # get training set size
    n_train = X_train_raw.shape[0]
    n_test = X_test_raw.shape[0]

    # normalize all features to [0,1]
    X_all = normalize_0_1(np.concatenate((X_train_raw, X_test_raw), axis=0))
    X_train = X_all[:n_train]
    X_test = X_all[n_train:]

    # convert each label into a 0-1 vector
    y_train = np.zeros((n_train, k))
    y_test = np.zeros((n_test, k))
    for i in range(n_train):
        y_train[i, int(y_train_raw[i])] = 1.0
    for i in range(n_test):
        y_test[i, int(y_test_raw[i])] = 1.0

    # Insure that the data correctly loaded in.
    assert X_train.shape == (60000, 784), (
        "Incorrect input, expected (60000, 784), found " + X_train.shape
    )
    assert y_train.shape == (60000, 10), (
        "Incorrect input, expected (60000, 10), found " + y_train.shape
    )
    assert X_test.shape == (10000, 784), (
        "Incorrect input, expected (10000, 784), found " + X_test.shape
    )
    assert y_test.shape == (10000, 10), (
        "Incorrect input, expected (10000, 10), found " + y_test.shape
    )

    return (X_train, y_train, X_test, y_test)


def load_weights(file="data/seeded_weights.npz"):
    container = np.load(file)
    weight_list = [container[key] for key in container]
    return weight_list


if __name__ == "__main__":
    main()
