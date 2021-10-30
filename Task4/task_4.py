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


def plot_loss(x, y):
    plt.plot(x, y)
    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


DIMENSIONS = 4

noise = lambda: np.random.normal(0, 10)  # some noise

ALPHA = 1e-4  # learning rate
TRAINING_STEPS = 10000

_train_size = len(X)


def calc_theta(LAMBDA=0.5):
    # define the computational graph
    graph = tf.Graph()
    with graph.as_default():
        # declare graph inputs
        x_train = tf.compat.v1.placeholder(tf.float32, shape=(_train_size, DIMENSIONS))
        y_train = tf.compat.v1.placeholder(tf.float32, shape=(_train_size, 1))
        theta = tf.Variable([[0.0] for _ in range(DIMENSIONS)])
        theta_0 = tf.Variable([[0.0]])  # don't forget the bias term!
        # forward propagation
        train_prediction = tf.matmul(x_train, theta) + theta_0
        # cost function and optimizer
        train_cost = (tf.nn.l2_loss(train_prediction - y_train) + LAMBDA * tf.matmul(tf.transpose(theta),
                                                                                     theta)) / float(
            _train_size)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(ALPHA).minimize(train_cost)

    # run the computation
    with tf.compat.v1.Session(graph=graph) as s:
        tf.compat.v1.initialize_all_variables().run()
        print("initialized");
        print(theta.eval())
        for step in range(TRAINING_STEPS):
            _, train_c = s.run([optimizer, train_cost], feed_dict={x_train: X, y_train: Y})
            if (step % 100 == 0):
                pass

        # PARAMETERS_GRADDESC = tf.concat(0, [theta_0, theta]).eval()
        # print("Solution for parameters:\n", PARAMETERS_GRADDESC)
        print("\nAfter", TRAINING_STEPS, "iterations:")
        print("   train cost =", train_c)
        return train_c


LAMBDAS = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.75, 0.9, 0.95, 0.99, 1, 2, 5, 10]
LOSSES = []

for lmbd in LAMBDAS:
    LOSSES.append(calc_theta(lmbd)[0][0])


plot_loss(LAMBDAS, LOSSES)
