'''
Created on Oct 7, 2015
@author: Ayush Jain 260674323
'''

import sys, os, codecs
reload(sys)
sys.setdefaultencoding('utf8')

import sklearn
import numpy as np
from nltk.corpus import stopwords
from sklearn import svm, linear_model, naive_bayes 
import nltk.data


######################################
# importing from different modoules
from dataParser import parseCSV
from featureSelector import extract_featureMatrix
from gaussian import Gaussian
from svmClassifier import SVM

######################################
# model parameters
m = 500
l = 50
lemmatize = False
lowercase = True
trainingDataPortion = 0.85

stoplist = stopwords.words('english')
stoplist.append('__eos__')

no_of_features=1500
replace = False

################################################################################
# reading data set and feature extraction
def read_data(filename, entriesToProcess):
    '''
    Read data set and return feature matrix X and class Y.
    X - (entriesToProcess x nfeats)
    Y - (entriesToProcess)
    '''
    directory =os.getcwd()
    os.chdir("Data")

    if (not replace) and os.path.isfile("featureMatrix.npy") and os.path.isfile("prediction.npy") : 
        print('Reading saved data from npy file')        
        X = np.load("featureMatrix.npy")
        Y = np.load("prediction.npy")

    else:
        os.chdir("..")
        interviews, Y = parseCSV(filename, entriesToProcess)
        entriesToProcess = Y.shape[0]
        X = extract_featureMatrix(interviews, Y, no_of_features, replace, lemmatize, lowercase, entriesToProcess)
        os.chdir("Data")
        np.save("featureMatrix",X)
        np.save("prediction", Y)
    
    os.chdir("..")
    return X, Y


################################################################################
# evaluation code
def accuracy(gold, predict):
    assert len(gold) == len(predict)
    assert len(gold) != 0 and len(predict) != 0
    corr = 0
    for i in xrange(len(gold)):
        if int(gold[i]) == int(predict[i]):
            corr += 1
    acc = float(corr) / len(gold)
    print ("Accuracy %d / %d = %.4f" % (corr, len(gold), acc))
################################################################################

if __name__ == '__main__':
	# main driver code
	
    filename = sys.argv[1]
    entriesToProcess = int(sys.argv[2])
    
    X, Y = read_data(filename, entriesToProcess)
    
    noOfTrainingEntries = int(Y.shape[0] * trainingDataPortion)
    
    train_X = X[:noOfTrainingEntries, :]
    train_Y = Y[:noOfTrainingEntries]
    test_X = X[noOfTrainingEntries:,:]
    test_Y = Y[noOfTrainingEntries:]
    
    #Naive Bayes
    print ("==============================================================================")
    print ("Naive Bayes Appproach")
    
    gnb = Gaussian(train_X, train_Y, 0.2)
    gnb.learn(train_X, train_Y)
    
    print ("Training Data Analysis:")
    predict = gnb.predict(train_X)
    accuracy(train_Y, predict)
    	
    print ("\nTesting Data Analysis:")
    predict = gnb.predict(test_X)	
    accuracy(test_Y, predict)
    
    #SVM
    print ("==============================================================================")
    print ("SVM Appproach")
    
    svm_classifier = SVM(train_X, train_Y, 0.2)
    svm_classifier.learn(train_X, train_Y)
    
    print ("Training Data Analysis:")
    predict = svm_classifier.predict(train_X)
    accuracy(train_Y, predict)
    
    print ("\nTesting Data Analysis:")
    predict = svm_classifier.predict(test_X)    
    accuracy(test_Y, predict)
    print ("==============================================================================")