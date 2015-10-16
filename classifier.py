'''
Created on Oct 7, 2015
@author: Ayush Jain 260674323
'''

import sys, os, codecs
reload(sys)
sys.setdefaultencoding('utf8')

import sklearn
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn import svm, linear_model, naive_bayes 
import nltk.data
from gaussian import Gaussian
from collections import Counter

######################################
# importing from different modoules
from dataParser import parseCSV
from featureSelector import get_bagofwords

######################################
# model parameters
m = 500
l = 50
lemmatize = False
lowercase = True
trainingDataPortion = 0.85

stoplist = set(stopwords.words('english'))


################################################################################
# reading data set
def read_data(filename, entriesToProcess):
	'''
	Read data set and return feature matrix X and class Y.
	X - (entriesToProcess x nfeats)
	Y - (entriesToProcess)
	'''
	interviews, Y = parseCSV(filename, entriesToProcess)

	#for i in range(entriesToProcess) :
	#	print interviews[i]
	
	X = extract_features(interviews, Y, m, l, lemmatize, lowercase, entriesToProcess)
    		
	return X, Y
    

################################################################################
# feature extraction

def extract_features(strings, classes, m, l, lemmatize, lowercase, entriesToProcess):
	'''
	Extract features from text file f into a feature vector.
    
	m: no. of most common ngrams to pick from each category
	l: no. of least common ngrams to pick from each category
	lemmatize: (boolean) whether or not to lemmatize
	lowercase: (boolean) whether or not to lowercase everything
	'''
		
	featurelist = get_bagofwords(strings, classes, m, l, lemmatize, lowercase)
	noOfFeatures = len(featurelist) # Total no. of features
	
	featureMatrix = np.zeros(shape=[entriesToProcess, noOfFeatures])
	i = 0
	for str in strings : 
		featureRow = {}
		bigram=[]
		tokens = get_tokens(str)
		for pair in nltk.bigrams(tokens) : bigram.append(pair)
		tokens.extend(bigram)
		
		dict_token = Counter(tokens)
		for inst in featurelist : 
			if (inst in tokens): 
				featureRow[inst] = dict_token[inst]
			else :
				featureRow[inst] = 0
		j=0
		for key, value in featureRow.items() :
			featureMatrix[i,j] = value
			j+=1
		i+=1
	
	return featureMatrix


	
def ispunct(some_string):
    return not any(char.isalnum() for char in some_string)

def get_tokens(s):
	'''
	Tokenize into words in sentences.
    
	Returns list of strs
	'''
	retval = []
	#sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	#sents = sent_detector.tokenize(s.strip())
	sents = sent_tokenize(s)
	for sent in sents:
		tokens = word_tokenize(sent)
		
		for word in tokens:
			word = word.lower()
			if (ispunct(word)): 
				continue
			if (word in stoplist): 
				continue
			elif (ispunct(word[-1])): 
				word = word[:-1] # removing puntuation from last place
			retval.append(word)
	
	return retval


################################################################################

# evaluation code
def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if int(gold[i]) == int(predict[i]):
            corr += 1
    acc = float(corr) / len(gold)
    print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)
################################################################################

if __name__ == '__main__':
	# main driver code
	
	filename = sys.argv[1]
	entriesToProcess = int(sys.argv[2])
	
	noOfTrainingEntries = int(entriesToProcess * trainingDataPortion)
	
	X, Y = read_data(filename, entriesToProcess)
	
	train_X = X[:noOfTrainingEntries, :]
	train_Y = Y[:noOfTrainingEntries]
	test_X = X[noOfTrainingEntries:,:]
	test_Y = Y[noOfTrainingEntries:]
	
	#Naive Bayes
	print "=============================================================================="
	print "Naive Bayes Appproach"
	
	gnb = Gaussian(train_X, train_Y, 0.2)
    gnb.learn(train_X, train_Y)
    
    print ("Training Data Analysis:")
    predict = gnb.predict(train_X)
    accuracy(train_Y, predict)
    	
    print ("\nTesting Data Analysis:")
    predict = gnb.predict(test_X)	
    print(predict)
    print(test_Y)
    accuracy(test_Y, predict)
	
	#SVM
	print "=============================================================================="
	print "SVM Appproach"
	
	clf = svm.SVC()
	clf.fit(train_X, train_Y)
	
	print "Training Data Analysis:"
	predict = clf.predict(train_X)
	accuracy(train_Y, predict)
	
	print "\nTesting Data Analysis:"
	predict = clf.predict(test_X)	
	accuracy(test_Y, predict)
	print "=============================================================================="