# -*- Coding: Utf-8 -*-
"""
created On Wed Oct 13 17:25:59 2015

@author: Neeth
"""

import Numpy As Np
#import Abc As Abcmeta

class Learner():#metaclass=abcmeta):
    """interface For Learning."""

    
    Def __init__(self, Train_x, Train_y, C_ratio):
        
        Self.x = Np.matrix(train_x)
        Self.y = Np.matrix(train_y)
        
        # Orient Y Matrix Correctly
        If Self.y.shape[0] != 1 Or Self.y.shape[1] > Self.y.shape[0]:
            Self.y = Self.y.t
            
        Self.n = Self.x.shape[0]
        Self.m = Self.x.shape[1]
        
        Self.train_len = 0
        Self.cross_len = 0
        Self.c_ratio   = C_ratio
            
        Self.train_x   = []
        Self.train_y   = []
        Self.c_valid_x = []
        Self.c_valid_y = []
    
        Self.c_indices = []

        Self.set_cross_validation_sets()
        
    Def Set_cross_validation_sets(self):
        If Self.c_ratio > 1 Or Self.c_ratio <= 0:
            Print('invalid C_ratio: ', Self.c_ratio)
            Self.c_indices = Np.matrix('-1, -1')
            Return
        
        Block_size = (int)(self.n * Self.c_ratio)
        
        If Block_size < 1:
            Print('cross-validation Block_size Is Less Than 1: ', Block_size)        
            Self.c_indices = Np.matrix('-1, -1')
            Return
        
        N_blocks = (int)(self.n / Block_size)
        
        If Self.n % Block_size != 0:
            N_blocks += 1
        
        Self.c_indices = Np.empty([n_blocks, 2])
        
        Self.c_indices[:, 0] = List(range(0, N_blocks * Block_size, Block_size))
        Self.c_indices[:, 1] = List(range(block_size, N_blocks * Block_size + 1, Block_size))
        
        Self.c_indices[-1, 1] = Self.n  
            
    Def Set_data(self, C_index):
        If Np.any(self.c_indices == -1):
            Self.train_len = Self.n
            Self.cross_len = 0
            
            Self.train_x = Self.x
            Self.train_y = Self.y
            
            Self.c_valid_x = Np.matrix('0, 0')
            Self.c_valid_y = Np.matrix('0, 0')
        Else:
            Lower = Self.c_indices[c_index][0]
            Upper = Self.c_indices[c_index][1]
            
            Self.cross_len = Upper - Lower
            Self.train_len = Self.n - Self.cross_len
            
            Self.c_valid_x = Self.x[lower:upper, :]
            Self.c_valid_y = Self.y[lower:upper, :]
            
            Self.train_x   = Np.empty([self.train_len, Self.m])
            Self.train_y   = Np.empty([self.train_len, 1])
            
            L_index = 0
            If Lower-1>= 0:
                Self.train_x[:lower, :] = Self.x[:lower, :]
                Self.train_y[:lower, :] = Self.y[:lower, :]
                L_index = Lower
            If Upper < Self.n:
                Self.train_x[l_index:, :] = Self.x[upper:self.n, :]
                Self.train_y[l_index:, :] = Self.y[upper:self.n, :]
            
    Def Do_kfold_cross_validation(self):
        Iter = List(range(self.c_indices.shape[0]))
        Cross_error = 0
        
        For I In Iter:
            Self.set_data(i)
            Weights = Self.learn();
            If Self.cross_len > 0:
                Cross_error += Self.calc_error(weights, Self.c_valid_x, Self.c_valid_y)
        
        Cross_error /= Iter[-1]
        Return Cross_error

#    @abstractmethod
    Def Learn(self):
        """to Be Defined In Child Classes."""
    
#    @abstractmethod
    Def Calc_error(self, Weights, X, Y):
        """to Be Defined In Child Classes."""
    
#    @abstractmethod
    Def Predict(self, Weights, X):
        """to Be Defined In Child Classes."""
    
    Def Tester(self):
        Self.do_kfold_cross_validation()
    
        
if __name__=='__main__':
    X = Np.matrix([[1, 2], [3, 4], [1, 2], [3, 4]])
    N = Np.matrix([1, 2, 1, 2])
   
    Lr = Learner(x, N, 0.25)
    Lr.tester();
    