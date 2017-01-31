#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:22:49 2017

@author: boris
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(path="data/"):
    """
    Load the movielens dataset and return a full data matrix with mask
    Also returns a list of (user, rated movie) couples (useful for splitting)
    """
    df = pd.read_csv(path + "ratings.csv")

    users = df.userId.unique()
    movies = df.movieId.unique()
    users.sort()
    movies.sort()

    u = len(users)
    m = len(movies)

    mask = np.zeros((u, m))
    data = np.zeros((u, m, 6))
    rated = []

    for j in tqdm(range(m)):
        tmp = df[df.movieId == movies[j]]
        users_tmp = tmp.userId.unique()
        for i in users_tmp:
            data[i - 1, j, tmp[tmp.userId == i].rating] = 1
            mask[i - 1, j] = 1
            rated.append((i - 1, j))

    return data, mask, rated


def train_test_split(data, mask, rated, test_pct=0.2):
    """
    """
    num_ratings = len(rated)
    num_test = int(num_ratings * test_pct)

    indices = np.random.choice(num_ratings, size=num_test, replace=False)
    test_set = np.asarray(rated)[indices]
    train_data = data.copy()
    train_mask = mask.copy()

    test_data = pd.DataFrame.from_records(test_set, columns=['userId', 'movieId'])
    test_data['rating'] = pd.Series(np.zeros(num_test), index=test_data.index)

    for (i, j) in tqdm(test_set):
        test_data[(test_data.userId == i) & (test_data.movieId == j)].rating = data[i, j, :].argmax()
        train_mask[i, j] = 0
        train_data[i, j] = np.zeros(6)

    return train_data, train_mask, test_data
