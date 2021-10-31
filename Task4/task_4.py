from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.disable_eager_execution()

time_start = time.time()
D = np.loadtxt('../data/lin_reg.txt', delimiter=',')

X = D[:, :-1]
Y = D[:, -1]
Y = np.reshape(Y, [1000, 1])


# График ошибки от параметра регуляризации
def plot_loss(x, y_1, y_2):
    plt.plot(x, y_1, label="Train")
    plt.plot(x, y_2, label="Test")
    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# График ошибки от эпохи
def plot_education(y_1, y_2):
    plt.plot(y_1, label="Train")
    plt.plot(y_2, label="Test")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# Сдвиг элементов массива на n
def shift(xs, n):
    m = len(xs) - n
    e = np.empty_like(xs)
    e[:m] = xs[n:]
    e[m:] = xs[:n]
    return e


DIMENSIONS = int(X.shape[1])
ALPHA = 1e-3
TRAINING_STEPS = 1000
TRAIN_RATION = 0.8

train_size = int(len(X) * TRAIN_RATION)
test_size = len(X) - train_size

train_X = X[:train_size, :]
train_Y = Y[:train_size, :]
test_X = X[train_size:, :]
test_Y = Y[train_size:, :]


# Обучение
def calc_theta(LAMBDA=0.5):
    graph = tf.Graph()
    with graph.as_default():
        x_train = tf.compat.v1.placeholder(tf.float32, shape=(train_size, DIMENSIONS))
        y_train = tf.compat.v1.placeholder(tf.float32, shape=(train_size, 1))
        x_test = tf.compat.v1.placeholder(tf.float32, shape=(test_size, DIMENSIONS))
        y_test = tf.compat.v1.placeholder(tf.float32, shape=(test_size, 1))
        theta = tf.Variable([[0.0] for _ in range(DIMENSIONS)])
        theta_0 = tf.Variable([[0.0]])

        train_prediction = tf.matmul(x_train, theta) + theta_0
        test_prediction = tf.matmul(x_test, theta) + theta_0

        train_cost = (tf.nn.l2_loss(train_prediction - y_train) + LAMBDA * tf.matmul(tf.transpose(theta),
                                                                                     theta)) / float(train_size)
        test_cost = (tf.nn.l2_loss(test_prediction - y_test) + LAMBDA * tf.matmul(tf.transpose(theta), theta)) / float(
            test_size)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(ALPHA).minimize(train_cost)

    time_start = time.time()
    with tf.compat.v1.Session(graph=graph) as s:
        tf.compat.v1.initialize_all_variables().run()
        print("Session starting in: " + str(time.time() - time_start))
        time_start = time.time()
        train_c_arr = []
        test_c_arr = []
        for step in range(TRAINING_STEPS):
            _, train_c, test_c = s.run([optimizer, train_cost, test_cost],
                                       feed_dict={x_train: train_X, y_train: train_Y, x_test: test_X, y_test: test_Y})
            train_c_arr.append(train_c[0][0])
            test_c_arr.append(test_c[0][0])

        print("Model fit in: " + str(time.time() - time_start))
        return train_c[0][0], test_c[0][0], train_c_arr, test_c_arr


# Нарисуйте график среднеквадратичной ошибки в зависимости от параметра регуляризации
def task_1():
    LAMBDAS = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.75, 0.9, 0.95, 0.99, 1, 2, 5, 10]
    LOSSES_TRAIN = []
    LOSSES_TEST = []
    for lmbd in LAMBDAS:
        train_l, test_l, _, _ = calc_theta(lmbd)
        LOSSES_TRAIN.append(train_l)
        LOSSES_TEST.append(test_l)

    plot_loss(LAMBDAS, LOSSES_TRAIN, LOSSES_TEST)


# Подготовьте исходные данные для 5 fold CV
def task_2(count, lmbd=0.5):
    global train_X, train_Y, test_X, test_Y
    TRAIN_RATION = (count - 1) / count

    train_size = int(len(X) * TRAIN_RATION)
    test_size = len(X) - train_size

    train_err = []
    test_err = []

    tmp_X = X.copy()
    tmp_Y = Y.copy()
    for i in range(count):
        tmp_X = shift(tmp_X, train_size)
        tmp_Y = shift(tmp_Y, train_size)
        train_X = tmp_X[:train_size, :]
        train_Y = tmp_Y[:train_size, :]
        test_X = tmp_X[train_size:, :]
        test_Y = tmp_Y[train_size:, :]
        tr_err, tst_err = calc_theta(lmbd)
        train_err.append(tr_err)
        test_err.append(tst_err)

    res_train = np.array(train_err)
    res_test = np.array(train_err)
    return res_train.mean(), res_test.mean()


# Поиск оптимального параметра регуляризации
# Вообще судя по графикам ошибка линейно зависит от LAMBDA и от сюда LAMBDA находится 0
# Но я думаю, что у меня тут ошибка
def task_3():
    min_lmbd = 0
    min_err = 1000
    for lmbd in np.linspace(0, 1, 50):
        err_a, err_b = task_2(5, lmbd)
        print("Last error is " + str(err_a))
        if err_a < min_err:
            min_err = err_a
            min_lmbd = lmbd
    return min_lmbd


# Кривая обучения
def task_4():
    train_err, test_err, train_arr, test_arr = calc_theta(0)
    plot_education(train_arr, test_arr)


task_4()
