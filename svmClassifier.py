# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc.common import logsumexp as spy
from learner import Learner
from sklearn import svm

class SVM(Learner):
    
    def __init__(self, train_X, train_Y, c_ratio):
        super(SVM, self).__init__(train_X, train_Y, c_ratio)
        self.cls = svm.LinearSVM

        
    def learn(self, x, y, c_valid = False):
        self.cls.fit(x,y)
            
    def predict(self, x):
        return self.cls.predict(x)
     
        
    def tester(self):
        x = np.matrix([[1, 2], [3, 4], [1, 2], [3, 4]])
        n = np.matrix([1, 0, 1, 0]).T
        
        self.learn(x, n)
        print(self.predict(x))
if __name__=='__main__':
    x = np.matrix([[1, 2], [3, 4], [10, 20], [30, 40]])
    n = np.array([1, 0, 1, 0]).T
   
    lr = Gaussian(x, n, 0.25)
    lr.learn(x, n)
    print('predict')
    print(lr.predict(x))