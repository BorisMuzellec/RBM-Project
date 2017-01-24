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
data = data[mnist.target == 0]
np.random.shuffle(data)

rbm = RBM(num_hidden=100, num_visible=784)
# display_weights(rbm.weights)

print("Training")
rbm.train(data, method="PCD", learning_rate=0.1, num_iter=100, k=5)

display_weights(rbm.weights)
_, samp = rbm.gibbs_vhv(data[0, :], k=5)
sb.heatmap(samp.reshape((28, 28)), cmap='gray')
plt.show()
