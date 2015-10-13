import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter



stoplist = set(stopwords.words('english'))


def ispunct(some_string):
    return not any(char.isalnum() for char in some_string)

	
def get_tokens(s):
#	Tokenize into words in sentences. Returns list of strs
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

	
def get_bagofwords(strings, classes, m, l, lemmatize, lowercase):
#	Returns the bag of most common n words for eah category
	featurelist=[]
	
	bow = []
	bow.append([])
	bow.append([])
	bow.append([])
	bow.append([])
	i = 0
	
	for str in strings : 
		tokens = get_tokens(str)
		
		unigram = tokens
		bigram = nltk.bigrams(tokens)
		category = int(classes[i])
		
		# Adding Unigrams to bag of words
		for inst in unigram : bow[category].append(inst)
		# Adding Bigrams to bag of words
		for inst in bigram : bow[category].append(inst)
		i += 1
	
	
	# Printing list of all n-grams for each prediction class
	#for j in range(3):
	#	print "Prediction Class : ",j, '\n', bow[j], '\n'
		
	# Picking best n-grams for each category and converting them to features
	for j in range(4):
		dict_ngram = Counter(bow[j])
	#	print dict_ngram
		
		for key, val in dict_ngram.most_common(m) : 
			featurelist.append(key)
			
		for key,val in dict_ngram.most_common()[-l-10-1:-10:1] : 
			featurelist.append(key)
	
	return featurelist