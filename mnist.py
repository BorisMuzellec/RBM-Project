import os

import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import seaborn as sb

from RBM import RBM


DATA_PATH = os.path.abspath(os.path.curdir)

print("Loading MNIST")
mnist = fetch_mldata('MNIST original', data_home=DATA_PATH)
data = mnist.data / 256.0 # Rescale data

# Shuffle the samples
idx = np.arange(mnist.data.shape[0])
np.random.shuffle(idx)
data = data[idx]
target = mnist.target[idx]


rbm = RBM(num_hidden=100, num_visible=784)

print("Training")
rbm.train(data[:10000], method="PCD", learning_rate=0.1, num_iter=1000, k=5)

sb.heatmap(rbm.weights[:, 10].reshape((28, 28)), cmap='gray')
plt.show()
sb.heatmap(rbm.weights[:, 50].reshape((28, 28)), cmap='gray')
plt.show()
sb.heatmap(rbm.weights[:, 100].reshape((28, 28)), cmap='gray')
plt.show()
sb.heatmap(rbm.weights[:, 250].reshape((28, 28)), cmap='gray')
plt.show()
