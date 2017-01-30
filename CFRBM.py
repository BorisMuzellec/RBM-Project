import numpy as np
from tqdm import tqdm
from utils import *


class CFRBM:
    def __init__(self, num_hidden, num_visible, num_rates):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.num_rates = num_rates
        # Break the symmetry
        self.weights = 0.05 * np.random.randn(self.num_visible, self.num_hidden, self.num_rates)
        self.visible_biases = 0.01 * np.random.randn(self.num_visible, self.num_rates)
        self.hidden_biases = 0.01 * np.random.randn(self.num_hidden)[np.newaxis, :]
        self.persistent_visible_ = None

    def _sample_hidden_probas(self, v):
        """
        Compute the probabilities that hidden units are activated, given visible units states
        """
        return sig(np.dot(v, self.weights) + self.hidden_biases)

    def _sample_visible_probas(self, h):
        """
        Compute the probabilities that visible units are activated, given hidden units states
        """
        # h_ = h[np.newaxis, :] if h.ndim == 1 else h
        # return sig(h_.dot(self.weights.T) + self.visible_biases)
        pass

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
        # return np.random.binomial(n=1, p=self._sample_visible_probas(h)) if binary else self._sample_visible_probas(h)
        pass

    def gibbs_vhv(self, v0, k=1, binary=True):
        """
        Perform k Gibbs sampling steps, starting from the visible units state v0
        """
        # v = v0
        # for t in range(k):
        #     h = self.sample_hidden(v)
        #     v = self.sample_visible(h, binary=True)
        # if not binary:
        #     v = self.sample_visible(h, binary)
        # return h, v
        pass

    def train(self):
        pass
