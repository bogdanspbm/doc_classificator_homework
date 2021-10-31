from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.disable_eager_execution()

D = np.loadtxt('../data/lin_reg.txt', delimiter=',')

X = D[:, :-1]
Y = D[:, -1]
Y = np.reshape(Y, [1000, 1])


def plot_loss(x, y_1, y_2):
    plt.plot(x, y_1, label="Train")
    plt.plot(x, y_2, label="Test")
    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


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


def calc_theta(LAMBDA=0.5):
    # define the computational graph
    graph = tf.Graph()
    with graph.as_default():
        # declare graph inputs
        x_train = tf.compat.v1.placeholder(tf.float32, shape=(train_size, DIMENSIONS))
        y_train = tf.compat.v1.placeholder(tf.float32, shape=(train_size, 1))
        x_test = tf.compat.v1.placeholder(tf.float32, shape=(test_size, DIMENSIONS))
        y_test = tf.compat.v1.placeholder(tf.float32, shape=(test_size, 1))
        theta = tf.Variable([[0.0] for _ in range(DIMENSIONS)])
        theta_0 = tf.Variable([[0.0]])  # don't forget the bias term!
        # forward propagation
        train_prediction = tf.matmul(x_train, theta) + theta_0
        test_prediction = tf.matmul(x_test, theta) + theta_0
        # cost function and optimizer
        train_cost = (tf.nn.l2_loss(train_prediction - y_train) + LAMBDA * tf.matmul(tf.transpose(theta),
                                                                                     theta)) / float(train_size)
        test_cost = (tf.nn.l2_loss(test_prediction - y_test) + LAMBDA * tf.matmul(tf.transpose(theta), theta)) / float(
            test_size)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(ALPHA).minimize(train_cost)

    # run the computation
    with tf.compat.v1.Session(graph=graph) as s:
        tf.compat.v1.initialize_all_variables().run()
        print("initialized");
        print(theta.eval())
        for step in range(TRAINING_STEPS):
            _, train_c, test_c = s.run([optimizer, train_cost, test_cost],
                                       feed_dict={x_train: train_X, y_train: train_Y, x_test: test_X, y_test: test_Y})
            if step % 100 == 0:
                pass

        # PARAMETERS_GRADDESC = tf.concat(0, [theta_0, theta]).eval()
        # print("Solution for parameters:\n", PARAMETERS_GRADDESC)
        print("\nAfter", TRAINING_STEPS, "iterations:")
        print("   train cost =", train_c)
        return train_c[0][0], test_c[0][0]


LAMBDAS = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.75, 0.9, 0.95, 0.99, 1, 2, 5, 10]
LOSSES_TRAIN = []
LOSSES_TEST = []
for lmbd in LAMBDAS:
    train_l, test_l = calc_theta(lmbd)
    LOSSES_TRAIN.append(train_l)
    LOSSES_TEST.append(test_l)

plot_loss(LAMBDAS, LOSSES_TRAIN, LOSSES_TEST)
