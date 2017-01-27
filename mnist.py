import os

import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import seaborn as sb

from RBM import RBM
from utils import display_weights


DATA_PATH = os.path.abspath(os.path.curdir)

print("Loading MNIST")
mnist = fetch_mldata('MNIST original', data_home=DATA_PATH)
data = mnist.data / 256.0  # Rescale data

# Shuffle the samples
# idx = np.arange(mnist.data.shape[0])
# np.random.shuffle(idx)
# data = data[idx]
# target = mnist.target[idx]

# We keep only the digit "0"
data = data[mnist.target == 2]
np.random.shuffle(data)
train_data = data[:6000, :]
validation_data = data[6000:, :]

rbm = RBM(num_hidden=50, num_visible=784)
n_iter = 200
k = 1

print("Training")
errors = np.zeros(n_iter)
rbm.train(train_data, method="PCD", learning_rate=0.1, num_iter=n_iter, k=k, errors=errors, decrease_eta=False)

# Plot the reconstruction error on the training set
# plt.figure()
# plt.plot(errors)

# Plot the weights of the RBM (one figure per hidden unit)
display_weights(rbm.weights, show=False)

# Plot one sample of visible units (based on hidden units computed from a real sample)
plt.figure()
_, samp = rbm.gibbs_vhv(validation_data[0, :], k=10, binary=False)
sb.heatmap(samp.reshape((28, 28)), cmap='gray')

plt.show()
