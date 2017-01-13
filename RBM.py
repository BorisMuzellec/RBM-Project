#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 20:00:30 2017

@author: boris
"""
import numpy as np
from tqdm import tqdm

def sig(x):
    """
    Computes the sigmoid of x in a vectorized way
    Args:
        x (np.ndarray): array to compute the sigmoid of
    """
    return 1/(1+np.exp(-x))


def iterate_minibatches(inputs, batchsize, shuffle=False):
    """Produces an batch iterator over the input.
    Usage:
        >>> for batch in iterate_minibatches(inputs, batchsize, shuffle):
        >>>    # do stuff
    Args:
        inputs (array-like): the input iterable over which iterate
        batchsize (int): the size of each batch (must be less than the number of inputs)
        shuffle (bool): if True, ``inputs`` is shuffled before iteration
    """
    N = inputs.shape[0]
    assert(N > batchsize)
    if shuffle:
        indices = np.arange(N)
        np.random.shuffle(indices)
    for start_idx in range(0, N - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


# NB: the input vectors should be fed as lines
class RBM:

    def __init__(self, num_hidden, num_visible):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        # Break the symmetry
        self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)
        self.visible_biases = np.zeros(num_visible)[np.newaxis, :]
        self.hidden_biases = np.zeros(num_hidden)[np.newaxis, :]

    def _sample_hidden_probas(self, v):
        """
        Compute the probabilities that hidden units are activated, given visible units states
        """
        v_ = v[np.newaxis, :] if v.ndim == 1 else v
        return sig(v_.dot(self.weights) + self.hidden_biases)

    def _sample_visible_probas(self, h):
        """
        Compute the probabilities that visible units are activated, given hidden units states
        """
        h_ = h[np.newaxis, :] if h.ndim == 1 else h
        return sig(h.dot(self.weights.T) + self.visible_biases)

    def sample_hidden(self, v):
        """
        Sample the hidden units state given visible units
        """
        p = self._sample_hidden_probas(v)
        return np.random.binomial(n = 1, p = p)

    def sample_visible(self, h, binary=False):
        """
        Sample the visible units state given hidden units
        """
        return np.random.binomial(n=1, p=self._sample_visible_probas(h)) if binary else self._sample_visible_probas(h)

    def gibbs_vhv(self, v0, k=1):
        """
        Perform k Gibbs sampling steps, starting from the visible units state v0
        """
        v = v0
        for t in range(k):
            h = self.sample_hidden(v)
            v = self.sample_visible(h)
        return h, v

    # Performs vanilla gradient ascent of the log-likelihood, using the prescribed method (only CD-k and PCD-k supported for now)
    def train(self, data, method="CD", learning_rate=0.1, batchsize=100, num_iter=100, k=10):
        for _ in tqdm(range(num_iter)):
            for batch in iterate_minibatches(data, batchsize=batchsize, shuffle=True):
                if method not in ["CD", "PCD"]:
                    raise NotImplementedError("Optimization method must be 'CD' or 'PCD'")

                W_grad, b_grad, c_grad = self._compute_grad(batch, k, method)
                self.weights += learning_rate * W_grad
                self.visible_biases += learning_rate * b_grad
                self.hidden_biases += learning_rate * c_grad

    # TODO: implement other training methods (Wasserstein...)
    def _compute_grad(self, batch, k, method="PCD"):
        N = batch.shape[0]

        if method == "CD":
            # Compute a Gibbs chains initialized with each sample in the batch
            v0 = batch
            _, v_tmp = self.gibbs_vhv(v0, k)
        elif method == "PCD":
            # We keep the visible states persistent between batches
            # If it is the first batch, we initilize the variable
            if not hasattr(self, 'persistent_visible_'):
                initial_hidden = np.zeros((N, self.num_hidden))
                self.persistent_visible_ = self.sample_visible(initial_hidden)

            # Gibbs sampling from persistant state
            v0 = self.persistent_visible_
            _, v_tmp = self.gibbs_vhv(self.persistent_visible_, k)
            # We keep this value for next batch
            self.persistent_visible_ = v_tmp

        # Compute the gradients for each chain (trick for W_grad, we already compute the sum of the gradients, see below)
        W_grad = (self._sample_hidden_probas(v0).T.dot(v0) - self._sample_hidden_probas(v_tmp).T.dot(v_tmp)).T
        b_grad = v0 - v_tmp
        c_grad = self._sample_hidden_probas(v0) - self._sample_hidden_probas(v_tmp)

        W_grad = W_grad / N # We just have to divise by the batch size
        b_grad = b_grad.mean(axis=0)[np.newaxis, :]
        c_grad = c_grad.mean(axis=0)[np.newaxis, :]

        return W_grad, b_grad, c_grad


    # Sample from the trained RBM (if visible = True, sample visible, else sample hidden)
    def sample(self, data, visible = True):
        if visible:
            return sig(self.weights.dot(data.T) + self.visible_biases[:, np.newaxis])
        else:
            p = sig(self.weights.T.dot(data.T) + self.hidden_biases[:, np.newaxis])
            return np.random.binomial(n = 1, p = p).T
