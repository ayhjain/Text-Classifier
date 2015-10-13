from sklearn import svm as SVM

class svm:
	# Class variables for svm
	def __init__(self):
		self.cls = SVM.SVC()
	
	def train(self, X, Y):
		self.cls.fit(X, Y)
	
	def predict(self, X) : 
		Y = self.cls.predict(X)
		return Y