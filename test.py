#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:56:27 2017

@author: boris
"""

import numpy as np
# import RBM
import CFRBM

# rbm = RBM.RBM(num_visible = 6, num_hidden = 2)
# training_data = np.array([[1,1,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
# rbm.train(training_data, method="PCD", batchsize=3)

# print("RBM weights:")
# print(rbm.weights)
# print("\n")

# print("Hidden layer biases:")
# print(rbm.hidden_biases)
# print("\n")

# print("Visible layer biases:")
# print(rbm.visible_biases)
# print("\n")

# print("Let's sample visible data from this distribution: ")
# hidden_data = np.array([[0,1],[1,1]])
# print(rbm.sample_visible(hidden_data, binary=True))
# print("\n")


# print("Let's sample hidden data from this distribution: ")
# visible_data = np.array([[0,1,0,0,1,1],[0,0,0,1,1,1]])
# print(rbm.sample_hidden(visible_data))

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
