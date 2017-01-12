#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 20:00:30 2017

@author: boris
"""
import numpy as np

def sig(x):
      return 1/(1+np.exp(-x))

#NB: the input vectors should be fed as lines
class RBM:
      
      def __init__(self, num_hidden, num_visible):
            self.num_hidden = num_hidden
            self.num_visible = num_visible
            #Break the symmetry
            self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)
            self.visible_biases = np.zeros(num_visible)
            self.hidden_biases = np.zeros(num_hidden)
      
      #Performs vanilla gradient ascent of the log-likelihood, using the prescribed method (only CD-k supported for now)
      def train(self, data, method = "CD", learning_rate = 0.1, num_iter = 100, k = 10):
            for i in range(num_iter):
                  W_grad, b_grad, c_grad = self._cd_grad(data, k)
                  self.weights += learning_rate * W_grad
                  self.visible_biases += learning_rate * b_grad
                  self.hidden_biases += learning_rate * c_grad
            
            
      #Implements the k-CD gradient sampling method
      #TODO: implement other training methods (PCD, Wasserstein...)
      def _cd_grad(self, data, k):
            W_grad = np.zeros(self.weights.shape)
            b_grad = np.zeros(self.visible_biases.shape)
            c_grad = np.zeros(self.hidden_biases.shape)
            
            for v in data:
                  v_tmp = v
                  for t in range(k):
                        p_hidden = sig(v_tmp.dot(self.weights)+self.hidden_biases)
                        #Little trick: Binomial(n=1) is the same as Bernoulli, but we can pass an array of probabilities as an argument
                        h = np.random.binomial(n=1, p = p_hidden)
                        p_visible = sig(self.weights.dot(h) + self.visible_biases)
                        #Same trick
                        v_tmp = np.random.binomial(n = 1, p = p_visible)
                  #Update gradients
                  W_grad += np.asarray((np.matrix(sig(v.dot(self.weights) + self.hidden_biases)).T.dot(np.matrix(v)) - np.matrix(sig(v_tmp.dot(self.weights) + self.hidden_biases)).T.dot(np.matrix(v_tmp))).T)
                  b_grad += v-v_tmp
                  c_grad += sig(v.dot(self.weights) + self.hidden_biases) - sig(v_tmp.dot(self.weights) + self.hidden_biases)
            
            return W_grad, b_grad, c_grad
            
      #Sample from the trained RBM (if visible = True, sample visible, else sample hidden)
      def sample(self, data, visible = True):
            p = sig(self.weights.dot(data) + self.visible_biases[:,np.newaxis]) if visible else  sig(data.dot(self.weights) + self.hidden_biases)
            return np.random.binomial(n = 1, p = p).T
                  
      
            
            