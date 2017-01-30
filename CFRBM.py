import numpy as np
from tqdm import tqdm
from utils import *




      
class CFRBM:
    """
    This implementation of RBMs for Collaborative Filtering
    only works when given one user input (of size #films x #ratings) at a time
    No vectorization over users for now
    """
    def __init__(self, num_hidden, num_visible, num_rates):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.num_rates = num_rates
        # Break the symmetry
        self.weights = {}
        self.visible_biases = {}
        for k in range (num_rates):
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
              acc += np.dot(v[:,k], self.weights[k])
        return sig(acc + self.hidden_biases)

    def _sample_visible_probas(self, h):
        """
        Compute the probabilities that visible units are activated, given hidden units states
        """
        h_ = h[np.newaxis, :] if h.ndim == 1 else h
        # return sig(h_.dot(self.weights.T) + self.visible_biases)
        v = np.zeros((self.num_visible, self.num_rates))
        for k in range(self.num_rates):
              v[:,k] = np.exp(h_.dot(self.weights[k].T) + self.visible_biases[k])
        return v/np.sum(v, axis = 1)[:,np.newaxis]

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
                    v[i,np.random.choice(self.num_rates, 1, p=probas[i])] = 1
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
       

    def train(self):
        pass
