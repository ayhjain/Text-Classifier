from multiprocessing import Pool, Queue
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path
from nltk.stem import WordNetLemmatizer
import string
import pickle


y = np.zeros(shape=[10,10])

def sq(x, y):
	i=0
	for i in range(x):
		y[i] = x**2
	

class LemmaTokenizer(object):
    def __init__(self,doc):
        self.wnl = WordNetLemmatizer()
        s = "".join(doc.split("__EOS__"))
        print s
        doc = s.translate(None, string.punctuation)
        tokens = word_tokenize(doc)
        bi = list(p1+" "+p2 for p1,p2 in nltk.bigrams(tokens))
        tokens.extend(bi)
        return [self.wnl.lemmatize(t) for t in tokens]   


if __name__=="__main__":
	s="   I touched my arm on the stove, and I got burned, and I cried.__EOS__  I bet you did. I bet that hurt. What happened? Did Daddy let you cook by yourself or something? Oh, God. OK. The court says that I'm not supposed to say anything, but I would be, like, totally negligent if I didn't tell you that you really have to watch yourself when you're alone with Daddy, OK?__EOS__  Margo was there.__EOS__  Margo? You mean our Margo? Margo who was at our house, that Margo? Do you have a nice room at Daddy's?"
	print (t for t in LemmaTokenizer(s))
	'''
	p=Pool(5)
	x=np.zeros(shape=[2,10,10])
	for i in range(10):
		x[i] = np.random.rand(10,10)
	p.map(sq, x)
	print y
	'''