import csv, sys
import numpy as np

def parseCSV(filename, entriesToProcess):
	
	x = []
	y = np.zeros(shape=[entriesToProcess, 1])
	
	with open(filename, 'rU') as inputs:
		reader = csv.DictReader(inputs)
		i = 0
		for row in reader :
			for key, value in row.items() :
#				print i, key, value
				if (key != 'Id') :
					try:
						if (key == 'Prediction') :
							y[i,0] = int(value)
						elif (key == 'Interview'): 
							x.append(value)
					except:
						print "ERROR: ", i, key, value
						# control shouldn't be in this portion of Code

			i +=  1
			if (i >= entriesToProcess): 
				break
	
	print ("Done parsing!")
	return x,y


if __name__ == "__main__" : 
	
	filename = sys.argv[1]
	entriesToProcess = int(sys.argv[2])
	
	x, y = parseCSV(filename, entriesToProcess)
	'''
	for i in range(entriesToProcess):
		print y[i]
		print
		
	print x
	'''
	print "Bye"