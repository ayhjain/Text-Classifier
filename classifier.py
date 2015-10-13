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
	
	featurelist = extract_features(interviews, Y, m, l, lemmatize, lowercase)
    		
	# convert indices to numpy array
	X = np.array(featurelist)
	return X, Y
    

################################################################################
# feature extraction

def extract_features(strings, classes, m, l, lemmatize, lowercase):
	'''
	Extract features from text file f into a feature vector.
    
	m: no. of most common ngrams to pick from each category
	l: no. of least common ngrams to pick from each category
	lemmatize: (boolean) whether or not to lemmatize
	lowercase: (boolean) whether or not to lowercase everything
	'''
	
	featurelist = get_bagofwords(strings, classes, m, l, lemmatize, lowercase)
		
	featureRow = []
	bigram=[]
	for str in strings : 
		bigram=[]
		tokens = get_tokens(str)
		for pair in nltk.bigrams(tokens) : bigram.append(pair)
		tokens.extend(bigram)
		
	for inst in tokens : 
		if (inst in featurelist): 
			print inst; 
			
	'''	
	for pair in global_bigrams : 
		if (pair in bigram):
			#print int(counter_bi[pair]); 
			indices[pair] = counter_bi[pair]
		else : indices[pair] = 0
    
	list=[]
	for value, key in indices.items() : 
		list.append(key)
	
	return list
	'''

	
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
	
	read_data(filename, entriesToProcess)
	
	print "done"
	'''
	i,j = train_X.shape
	sum=0
	for i1 in range(i):
		for i2 in range(j):
			sum+=train_X[i1,i2]
	print sum
	
	test_X, test_Y = read_tac('2011', True)
	
	
	#Naive Bayes
	print "====================================================================================="
	print "Naive Bayes Appproach"
	
	nb = naive_bayes.GaussianNB()
	nb.fit(train_X, train_Y)
	
	print "Training Data Analysis:"
	predict = nb.predict(train_X)
	accuracy(train_Y, predict)
	
	print "\nTesting Data Analysis:"
	predict = nb.predict(test_X)	
	accuracy(test_Y, predict)
	print "====================================================================================="
	
	#SVM
	print "====================================================================================="
	print "SVM Appproach"
	
	clf = svm.SVC()
	clf.fit(train_X, train_Y)
	
	print "Training Data Analysis:"
	predict = clf.predict(train_X)
	accuracy(train_Y, predict)
	
	print "\nTesting Data Analysis:"
	predict = clf.predict(test_X)	
	accuracy(test_Y, predict)
	print "====================================================================================="

	#Linear Model
	print "====================================================================================="
	print "Linear Model Appproach"
	
	clf = linear_model.SGDClassifier()
	clf.fit(train_X, train_Y)
	
	print "Training Data Analysis:"
	predict = clf.predict(train_X)
	accuracy(train_Y, predict)
	
	print "\nTesting Data Analysis:"
	predict = clf.predict(test_X)	
	accuracy(test_Y, predict)
	print "====================================================================================="
'''