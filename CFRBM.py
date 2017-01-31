import numpy as np
from tqdm import tqdm
from utils import *


def mse(ratings, preds):
    """
    Mean square error between predicted and actual ratings
    """
    return np.mean((ratings - preds) ** 2)


class CFRBM:
    """This implementation of RBMs for Collaborative Filtering
    only works when given one user input (of size #films x #ratings) at a time
    No vectorization over users for now"""

    def __init__(self, num_hidden, num_visible, num_rates):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.num_rates = num_rates
        # Break the symmetry
        self.weights = {}
        self.visible_biases = {}
        for k in range(num_rates):
            self.weights[k] = 0.05 * np.random.randn(self.num_visible, self.num_hidden)
            self.visible_biases[k] = 0.01 * np.random.randn(self.num_visible)
        self.hidden_biases = 0.01 * np.random.randn(self.num_hidden)[np.newaxis, :]
        self.persistent_visible_ = None

    def _sample_hidden_probas(self, v):
        """
        Compute the probabilities that hidden units are activated, given visible units states
        Assume that v is a one-hot vector (i.e. v[i,k] = 1 iff the rating of movie i is k)
        """
        acc = np.zeros(self.num_hidden)
        for k in range(self.num_rates):
            acc += np.dot(v[:, k], self.weights[k])
        return sig(acc + self.hidden_biases)

    def _sample_visible_probas(self, h):
        """
        Compute the probabilities that visible units are activated, given hidden units states
        """
        h_ = h[np.newaxis, :] if h.ndim == 1 else h
        # return sig(h_.dot(self.weights.T) + self.visible_biases)
        v = np.zeros((self.num_visible, self.num_rates))
        for k in range(self.num_rates):
            v[:, k] = np.exp(h_.dot(self.weights[k].T) + self.visible_biases[k])
        return v / np.sum(v, axis=1)[:, np.newaxis]

    def sample_hidden(self, v):
        """
        Sample the hidden units state given visible units
        """
        p = self._sample_hidden_probas(v)
        return np.random.binomial(n=1, p=p)

    def sample_visible(self, h, binary=False):
        """
        Sample the visible units state given hidden units
        """
        probas = self._sample_visible_probas(h)

        if binary:
            v = np.zeros(probas.shape)
            for i in range(probas.shape[0]):
                v[i, np.random.choice(self.num_rates, 1, p=probas[i])] = 1
        else:
            v = probas
        return v

    def gibbs_vhv(self, v0, k=1, binary=True):
        """
        Perform k Gibbs sampling steps, starting from the visible units state v0
        """
        v = v0
        for t in range(k):
            h = self.sample_hidden(v)
            v = self.sample_visible(h, binary=True)
        if not binary:
            v = self.sample_visible(h, binary)
        return h, v

    # Performs vanilla gradient ascent of the log-likelihood, using the prescribed method
    # Mask is a #user x m size matrix with 1 if the movie j was rated by user i, else 0
    # Assumes that films for which no rating was given are encoded as vectors full of zeroes
    def train(self, data, mask, method="CD", learning_rate=0.01, weight_decay=0.01, num_iter=100, k=10, decrease_eta=False):
        threshold = int(4 / 5 * num_iter)  # completely arbitrary choice
        N = data.shape[0]
        np.random.shuffle(data)
        for i in tqdm(range(num_iter)):
            if decrease_eta:
                # Decrease the learning rate after some iterations
                eta = learning_rate if i < threshold else learning_rate / ((1 + i - threshold) ** 2)
            else:
                eta = learning_rate

            for j in tqdm(range(N)):
                if method not in ["CD", "PCD"]:
                    raise NotImplementedError("Optimization method must be 'CD' or 'PCD'")

                W_grad, b_grad, c_grad = self._compute_grad(data[j], mask[j], k, method)
                for k in range(self.num_rates):
                    self.weights[k] = np.multiply(self.weights[k], (1 - weight_decay) * mask[j][:, np.newaxis]) + eta * W_grad[k]
                    self.visible_biases[k] = np.multiply(self.visible_biases[k], (1 - weight_decay) * mask[j]) + eta * b_grad[k]
                self.hidden_biases = (1 - weight_decay) * self.hidden_biases + eta * c_grad

                self.persistent_visible_ = None  # reset for each batch!

    def _compute_grad(self, v0, mask, k, method="PCD"):
        if method == "CD":
            # Compute a Gibbs chains initialized with each sample in the batch
            _, v_tmp = self.gibbs_vhv(v0, k)
        elif method == "PCD":
            # We keep the visible states persistent between batches
            # If it is the first batch, we initilize the variable
            if self.persistent_visible_ is None:
                self.persistent_visible_ = v0

            # Gibbs sampling from persistant state
            v0 = self.persistent_visible_
            _, v_tmp = self.gibbs_vhv(self.persistent_visible_, k)
            # We keep this value for next batch
            self.persistent_visible_ = v_tmp

        # Compute the gradients for each chain (trick for W_grad, we already compute the sum of the gradients, see below)
        W_grad = {}
        b_grad = {}
        for k in range(self.num_rates):
            W_grad[k] = np.multiply((self._sample_hidden_probas(v0).T.dot(v0[:, k][np.newaxis, :]) - self._sample_hidden_probas(v_tmp).T.dot(v_tmp[:, k][np.newaxis, :])).T, mask[:, np.newaxis])
            b_grad[k] = np.multiply(v0[:, k] - v_tmp[:, k], mask)
        c_grad = self._sample_hidden_probas(v0) - self._sample_hidden_probas(v_tmp)
        # print(np.linalg.norm(W_grad - W_grad[:, 0][:, np.newaxis]))

        return W_grad, b_grad, c_grad

    def predict_rating(self, v, q):
        """
        Given preferences v, predict rating for movie q
        """
        p_h = self._sample_hidden_probas(v)
        p_v = self._sample_visible_probas(p_h)

        return p_v[q, :].dot(np.arange(self.num_rates))
