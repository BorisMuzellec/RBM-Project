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


rbm = RBM(num_hidden=25, num_visible=784)

print("Training")
rbm.train(data, method="PCD", learning_rate=0.1, num_iter=10, k=1)

res = rbm.sample_visible(rbm.sample_hidden(data[:4]))
for r in res:
    sb.heatmap(r.reshape((28, 28)), cmap='gray')
    plt.show()
