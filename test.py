#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:56:27 2017

@author: boris
"""

import numpy as np
import RBM

rbm = RBM.RBM(num_visible = 6, num_hidden = 2)
training_data = np.array([[1,1,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
rbm.train(training_data)

print("RBM weights:")
print(rbm.weights)
print("\n")

print("Hidden layer biases:")
print(rbm.hidden_biases)
print("\n")

print("Visible layer biases:")
print(rbm.visible_biases)
print("\n")

print("Let's sample visible data from this distribution: ")
hidden_data = np.array([[0,1],[1,1]])
print(rbm.sample(hidden_data))
print("\n")


print("Let's sample hidden data from this distribution: ")
visible_data = np.array([[0,1,0,0,1,1],[0,0,0,1,1,1]])
print(rbm.sample(visible_data, visible = False))




