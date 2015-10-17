import csv, sys
import numpy as np

def parseCSV(filename, entriesToProcess):
    
    x = []
    y = []#np.zeros(entriesToProcess)
    
    with open(filename, 'rU') as inputs:
        reader = csv.DictReader(inputs)
        i = 0
        for row in reader :
            try:
                y.append(row['Prediction'])
                x.append(row['Interview'])
            except:
                print "Error parsing ", i,"th entry"
                # control shouldn't be in this portion of Code
            i +=  1
            if (i >= entriesToProcess and entriesToProcess >= 0):
                break

    print ("Done parsing!")
    y = np.array(y)
    return x,y


if __name__ == "__main__" : 
    
    filename = sys.argv[1]
    entriesToProcess = int(sys.argv[2])
    
    x, y = parseCSV(filename, entriesToProcess)
    print len(x), len(y)    
    print "Bye"