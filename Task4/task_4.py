import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.disable_eager_execution()
rng = np.random

print(tf.__version__)

D = np.loadtxt('../data/lin_reg.txt', delimiter=',')
LAMBDA = 0.5  # L2 regularization factor

X = D[:, :-1]
Y = D[:, -1]

learning_rate = 0.0001
training_epochs = 1000
display_step = 50

train_X = X.copy()
train_Y = Y.copy()

W = tf.Variable(rng.randn(4, 1), name="weight")
b = tf.Variable(rng.randn(1), name="bias")
LAMBDA = tf.Variable(0.5, name="regularization", dtype=tf.float64)

print(W)
print(b)

X = tf.compat.v1.placeholder(tf.float64, shape=(4))
Y = tf.compat.v1.placeholder(tf.float64, shape=())

pred = tf.add(tf.matmul(tf.reshape(X, shape=[1, 4]), W), b, name="prediction")

cost = tf.add(tf.reduce_sum(tf.pow(pred - Y, 2)), tf.multiply(LAMBDA, tf.matmul(tf.transpose(W), W)), name="cost")

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.compat.v1.global_variables_initializer()


with tf.compat.v1.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: np.transpose(train_X), Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))
