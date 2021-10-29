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

X = tf.compat.v1.placeholder(tf.float64, shape=(1, 4))
Y = tf.compat.v1.placeholder(tf.float64, shape=(1))

pred = tf.add(tf.matmul(X, W), b, name="prediction")

cost = tf.add(tf.reduce_sum(tf.pow(pred - Y, 2)), tf.multiply(LAMBDA, tf.matmul(tf.transpose(W), W)), name="cost")

print(cost)
