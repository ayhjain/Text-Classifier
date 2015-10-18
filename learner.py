# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:25:59 2015

@author: neeth
"""

import numpy as np
#import abc as ABCMeta

class Learner(object):#metaclass=ABCMeta):
    """Interface for learning."""

    
    def __init__(self, train_X, train_Y, c_ratio):
        
        self.x = np.ones([train_X.shape[0], train_X.shape[1]+1])
        self.x[:, 1:] = np.matrix(train_X)
        self.y = np.matrix(train_Y)
        
        # orient y matrix correctly
        if self.y.shape[1] != 1 or self.y.shape[1] > self.y.shape[0]:
            self.y = self.y.T
            
        self.n = self.x.shape[0]
        self.m = self.x.shape[1]
        
        self.train_len = 0
        self.cross_len = 0
        self.c_ratio   = c_ratio
            
        self.train_X   = []
        self.train_Y   = []
        self.c_valid_X = []
        self.c_valid_Y = []
    
        self.c_indices = []

        self.set_cross_validation_sets()
        
    def set_cross_validation_sets(self):
        if self.c_ratio > 1 or self.c_ratio <= 0:
            print('invalid c_ratio: ', self.c_ratio)
            self.c_indices = np.matrix('-1, -1')
            return
        
        block_size = (int)(self.n * self.c_ratio)
        
        if block_size < 1:
            print('cross-validation block_size is less than 1: ', block_size)        
            self.c_indices = np.matrix('-1, -1')
            return
        
        n_blocks = (int)(self.n / block_size)
        
        if self.n % block_size != 0:
            n_blocks += 1
        
        self.c_indices = np.empty([n_blocks, 2])
        
        self.c_indices[:, 0] = list(range(0, n_blocks * block_size, block_size))
        self.c_indices[:, 1] = list(range(block_size, n_blocks * block_size + 1, block_size))
        
        self.c_indices[-1, 1] = self.n  
            
    def set_data(self, c_index):
        if np.any(self.c_indices == -1):
            self.train_len = self.n
            self.cross_len = 0
            
            self.train_X = self.x
            self.train_Y = self.y
            
            self.c_valid_X = np.matrix('0, 0')
            self.c_valid_Y = np.matrix('0, 0')
        else:
            lower = self.c_indices[c_index][0]
            upper = self.c_indices[c_index][1]
            
            self.cross_len = upper - lower
            self.train_len = self.n - self.cross_len
            
            self.c_valid_X = self.x[lower:upper, :]
            self.c_valid_Y = self.y[lower:upper, :]
            
            self.train_X   = np.empty([self.train_len, self.m])
            self.train_Y   = np.empty([self.train_len, 1])
            
            l_index = 0
            if lower-1>= 0:
                self.train_X[:lower, :] = self.x[:lower, :]
                self.train_Y[:lower, :] = self.y[:lower, :]
                l_index = lower
            if upper < self.n:
                self.train_X[l_index:, :] = self.x[upper:self.n, :]
                self.train_Y[l_index:, :] = self.y[upper:self.n, :]
            
    def do_kfold_cross_validation(self):
        iter = list(range(self.c_indices.shape[0]))
        cross_error = 0
        
        for i in iter:
            self.set_data(i)
            weights = self.learn(self.train_X, self.train_Y, c_valid = True);
            if self.cross_len > 0:
                cross_error += self.calc_error(weights, self.c_valid_X, self.c_valid_Y)
        
        cross_error /= iter[-1]
        return cross_error

#    @abstractmethod
    def learn(self, x, y, c_valid = False):
        """To be defined in child classes."""
    
#    @abstractmethod
    def calc_error(self, weights, x, y):
        """To be defined in child classes."""
    
#    @abstractmethod
    def predict(self, weights, x):
        """To be defined in child classes."""
    
    def tester(self):
#        self.do_kfold_cross_validation()
        pass
    
if __name__=='__main__':
    x = np.matrix([[1, 2], [3, 4], [1, 2], [3, 4]])
    n = np.matrix([1, 0, 1, 0]).T
   
    lr = Learner(x, n, 0.25)
    