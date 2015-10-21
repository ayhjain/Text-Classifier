from multiprocessing import Pool, Queue
import numpy as np


y = np.zeros(shape=[10,10])

def sq(x, y):
	i=0
	for i in range(x):
		y[i] = x**2
	


if __name__=="__main__":
	p=Pool(5)
	x=np.zeros(shape=[2,10,10])
	for i in range(10):
		x[i] = np.random.rand(10,10)
	p.map(sq, x)
	print y
