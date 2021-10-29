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

DIMENSIONS = 4

noise = lambda: np.random.normal(0, 10)  # some noise

ALPHA = 1e-6  # learning rate
LAMBDA = 0.5  # L2 regularization factor
TRAINING_STEPS = 10000

_train_size = len(X)

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
    train_cost = (tf.nn.l2_loss(train_prediction - y_train) + LAMBDA * tf.nn.l2_loss(theta)) / float(_train_size)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(ALPHA).minimize(train_cost)

# run the computation
with tf.compat.v1.Session(graph=graph) as s:
    tf.compat.v1.initialize_all_variables().run()
    print("initialized");
    print(theta.eval())
    for step in range(TRAINING_STEPS):
        _, train_c = s.run([optimizer, train_cost], feed_dict={x_train: X, y_train: Y})
        if (step % 100 == 0):
            print("\nAfter", step, "iterations:")
            print("   train cost =", train_c)

    # PARAMETERS_GRADDESC = tf.concat(0, [theta_0, theta]).eval()
    # print("Solution for parameters:\n", PARAMETERS_GRADDESC)
