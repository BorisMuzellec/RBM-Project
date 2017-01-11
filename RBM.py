#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 20:00:30 2017

@author: boris
"""
import numpy as np

def sig(x):
      return 1/(1+np.exp(-x))

class RBM:
      
      def __init__(self, num_hidden, num_visible):
            self.num_hidden = num_hidden
            self.num_visible = num_visible
            self.weights = np.zeros((num_hidden, num_visible))
            self.visible_biases = np.zeros(num_hidden)
            self.hidden_biases = np.zeros(num_visible)
      
      #Performs vanilla gradient ascent of the log-likelihood, using the prescribed method (only CD-k supported for now)
      def train(self, data, method = "CD", learning_rate = 0.1, num_iter = 100, k = 10):
            for i in range(num_iter):
                  W_grad, b_grad, c_grad = _cd_grad(self, data, k)
                  self.weights += learning_rate * W_grad
                  self.visible_biases += learning_rate * b_grad
                  self.hidden_biases += learning_rate * c_grad
            
            
      #Implements the k-CD gradient sampling method
      def _cd_grad(self, data, k):
            W_grad = np.zeros(self.weights.shape)
            b_grad = np.zeros(self.visible_biases.shape)
            c_grad = np.zeros(self.hidden_biases.shape)
            
            for v in data:
                  v_tmp = v
                  for t in range(k):
                        p_hidden = sig(self.weights.dot(v_tmp)+self.visible_biases)
                        #Little trick: Binomial(n=1) is the same as Bernoulli, but we can pass an array of probabilities as an argument
                        h = np.random.binomial(n=1, p = p_hidden)
                        p_visible = sig(self.weights.T.dot(h) + self.hidden_biases)
                        #Same trick
                        v_tmp = np.random.binomial(n = 1, p_visible)
                  #Update gradients
                  W_grad += np.asarray(np.matrix(sig(self.weight.dot(v) + c)).T.dot(np.matrix(v)) - np.matrix(sig(self.weight.dot(v_tmp) + c)).T.dot(np.matrix(v_tmp)))
                  b_grad += v-v_tmp
                  c_grad += sig(self.weight.dot(v) + c) - sig(self.weight.dot(v_tmp) + c)
            
            return W_grad, b_grad, c_grad
                  
      