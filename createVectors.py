
#from ass4 import create_dataset
def returnVectors(a,d,length):
	X=[]
	
	for i in range(length):
		p=a+(i-1)*d
		X.append(p)

	print(X)

	Y = X[length-2]+d
	print(Y)
	return X,Y