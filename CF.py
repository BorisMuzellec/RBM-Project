#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:08:38 2017

@author: boris
"""
import numpy as np
import CFRBM
from movielens import *


try:
    data
    mask
    rated
except NameError:
    print("Loading data\n")
    data, mask, rated = load_data()
else:
    print("Data already in memory\n")

try:
    train_data
    train_mask
    test_data
except NameError:
    print("Split data\n")
    train_data, train_mask, test_data = train_test_split(data, mask, rated)
else:
    print("Data already splitted\n")

rbm = CFRBM.CFRBM(num_hidden=50, num_visible=train_data.shape[1], num_rates=6)

print("Train RBM\n")
rbm.train(train_data, train_mask, method="CD", learning_rate=0.1, weight_decay=0.01, num_iter=1, k=5, decrease_eta=False)

print("Make predictions\n")
test_pairs = test_data['userId', 'movieId']
preds = []
ratings = test_data['rating'].values()

for (i, j) in test_pairs.values():
    preds.append(rbm.predict_rating(data[i, :], j))

print("Mean square loss: %f" % CFRBM.mse(ratings, preds))
