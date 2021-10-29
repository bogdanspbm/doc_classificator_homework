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


DIMENSIONS = 4

noise = lambda: np.random.normal(0,10) # some noise

####################################################################################################
### GRADIENT DESCENT APPROACH
####################################################################################################
# dataset globals
DS_SIZE = 5000
TRAIN_RATIO = 0.6  # 60% of the dataset is used for training
_train_size = int(DS_SIZE * TRAIN_RATIO)
_test_size = DS_SIZE - _train_size
ALPHA = 1e-8  # learning rate
LAMBDA = 0.5  # L2 regularization factor
TRAINING_STEPS = 1000

# generate the dataset, the labels and split into train/test
ds = [[np.random.rand() * 1000 for d in range(DIMENSIONS)] for _ in range(DS_SIZE)]  # synthesize data

ds = [(x, [sum(x)+noise()]) for x in ds]

np.random.shuffle(ds)

train_data, train_labels = zip(*ds[0:_train_size])
test_data, test_labels = zip(*ds[_train_size:])
print(np.asarray(train_data).shape, X.shape)
print(np.asarray(train_labels).shape, Y.shape)

# define the computational graph
graph = tf.Graph()
with graph.as_default():
    # declare graph inputs
    x_train = tf.compat.v1.placeholder(tf.float32, shape=(_train_size, DIMENSIONS))
    y_train = tf.compat.v1.placeholder(tf.float32, shape=(_train_size, 1))
    x_test = tf.compat.v1.placeholder(tf.float32, shape=(_test_size, DIMENSIONS))
    y_test = tf.compat.v1.placeholder(tf.float32, shape=(_test_size, 1))
    theta = tf.Variable([[0.0] for _ in range(DIMENSIONS)])
    theta_0 = tf.Variable([[0.0]])  # don't forget the bias term!
    # forward propagation
    train_prediction = tf.matmul(x_train, theta) + theta_0
    test_prediction = tf.matmul(x_test, theta) + theta_0
    # cost function and optimizer
    train_cost = (tf.nn.l2_loss(train_prediction - y_train) + LAMBDA * tf.nn.l2_loss(theta)) / float(_train_size)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(ALPHA).minimize(train_cost)
    # test results
    test_cost = (tf.nn.l2_loss(test_prediction - y_test) + LAMBDA * tf.nn.l2_loss(theta)) / float(_test_size)

# run the computation
with tf.compat.v1.Session(graph=graph) as s:
    tf.compat.v1.initialize_all_variables().run()
    print("initialized");
    print(theta.eval())
    for step in range(TRAINING_STEPS):
        _, train_c, test_c = s.run([optimizer, train_cost, test_cost],
                                   feed_dict={x_train: train_data, y_train: train_labels,
                                              x_test: test_data, y_test: test_labels})
        if (step % 100 == 0):
            # it should return bias close to zero and parameters all close to 1 (see definition of f)
            print("\nAfter", step, "iterations:")
            # print("   Bias =", theta_0.eval(), ", Weights = ", theta.eval())
            print("   train cost =", train_c);
            print("   test cost =", test_c)

    #PARAMETERS_GRADDESC = tf.concat(0, [theta_0, theta]).eval()
    #print("Solution for parameters:\n", PARAMETERS_GRADDESC)
