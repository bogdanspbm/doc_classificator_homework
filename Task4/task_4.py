import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

D = np.loadtxt('../data/lin_reg.txt', delimiter=',')
LAMBDA = 0.5  # L2 regularization factor

X = D[:, :-1]
Y = D[:, -1]

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X))

tf.keras.regularizers.L2(l2=1)

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    X,
    Y,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history)
