#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:56:27 2017

@author: boris
"""

import numpy as np
# import RBM
import CFRBM


rbm = CFRBM.CFRBM(num_hidden=2, num_visible=6, num_rates=3)
training_data = np.random.binomial(n=1, p=0.3, size=(10, 6, 3))  # 10 samples of size 6 * 3

print("RBM weights:")
print(rbm.weights)
print("\n")

print("Hidden layer biases:")
print(rbm.hidden_biases)
print("\n")

print("Visible layer biases:")
print(rbm.visible_biases)
print("\n")

print(rbm._sample_hidden_probas(training_data[2,:,:]))

h = rbm.sample_hidden(training_data[2,:,:])

print(rbm._sample_visible_probas(h))

print(rbm.sample_visible(h, binary = True))

_, samp = rbm.gibbs_vhv(training_data[2,:,:], k=3, binary = True)

print(samp)

rbm.train(training_data, method="PCD", learning_rate=0.1, num_iter=3, k=3, decrease_eta=False)

_, samp = rbm.gibbs_vhv(training_data[2,:,:], k=3, binary = True)

print(samp)

