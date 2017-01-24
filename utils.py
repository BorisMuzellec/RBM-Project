import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def sig(x):
    """
    Computes the sigmoid of x in a vectorized way
    Args:
        x (np.ndarray): array to compute the sigmoid of
    """
    return 1 / (1 + np.exp(-x))


def display_weights(weights, dimx=28, dimy=28, show=True):
    m, n = weights.shape
    cols = int(np.floor(np.sqrt(n)))
    rows = int(n / cols) if n % cols == 0 else int(np.floor(n / cols)) + 1
    r = n % rows
    fig, ax = plt.subplots(rows, cols, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.01, hspace=0.1)
    for i in range(rows):
        for j in range(cols):
            if r > 0 and i == (rows - 1) and j >= (r - 1):
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].set_axis_bgcolor('white')
            else:
                sb.heatmap(weights[:, i * cols + j].reshape((dimx, dimy)), cmap='gray', ax=ax[i, j], square=True, cbar=False, xticklabels=False, yticklabels=False)
    if show:
        plt.show()
