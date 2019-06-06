import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import pickle
from keras.layers import Flatten,Dropout,Activation
from keras.preprocessing.sequence import pad_sequences
from createVectors import returnVectors

np.random.seed(7)
# convert an array of values into a dataset mat
seq=[]
k=[]
def ap(a,n,d):
	for i in range(n):
		p=a+(i-1)*d
		seq.append(p)
	return seq
k=np.asarray(ap(0,100,5))

#scaler = MinMaxScaler(feature_range=(0, 1))
#k = scaler.fit_transform(k)

print(k.shape)	

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=10):
	dataX, dataY = [], []
	#k=np.asarray(ap(2,100,5))
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

k = np.reshape(k,(k.shape[0],1))















#[m,n]=create_dataset(k, look_back=6)

#print(m)
#print(m.shape)
#print(create_dataset(k, look_back=5))
# split into train and test sets
train_size = int(len(k) * 0.67)
test_size = len(k) - train_size
train, test = k[0:train_size], k[train_size:len(k)]

trainX, trainY = create_dataset(train, look_back=10)
testX, testY = create_dataset(test, look_back=10)

trainX=np.reshape(trainX,(trainX.shape[0],trainX.shape[1],1)).astype('float32')
testX=np.reshape(testX,(testX.shape[0],testX.shape[1],1)).astype('float32')
trainY = np.reshape(trainY,(trainY.shape[0],1)).astype('float32')
testY = np.reshape(testY,(testY.shape[0],1)).astype('float32')

trainX=trainX/5
testX=testX/5
trainY=trainY/5
testY=testY/5


#print(trainX)
#print(trainY)
#print(testX)
#print(testY)




#pk = pickle.load('/home/supriyo/practice/Assignment4/ap_5.pkl')
pickle_in = open("/home/supriyo/practice/Assignment4/ap/ap_4.pkl","rb")
example_dict = pickle.load(pickle_in)
#print(type(example_dict))
pk=np.asarray(example_dict)

print("********************************************")
print(pk)
print(pk.shape)

first = pk[0]
print(first[0][0])



length = 10

X = []
Y =[]

for i in range(10):
	first = pk[i]
	a=first[0][0]
	chotaX,chotaY = returnVectors(a,4,length)
	X.append(chotaX)
	Y.append(chotaY)


X=np.asarray(X)

Y=np.asarray(Y)

print("HOGYA SHAYAD")
print(X.shape)
print(Y.shape)
X=np.reshape(X,(X.shape[0],X.shape[1],1)).astype('float32')
Y = np.reshape(Y,(Y.shape[0],1)).astype('float32')
#X=X/300
#Y=Y/300






test2=[]

for i in range(10):
	test1=np.asarray(pk[i,0])
	test2.append(test1)

test2=np.asarray(test2)
#print(test2[0])	

test3=np.asarray(pk[:,1])
#print(test3)


EPOCHS = 10
INIT_LR = 1e-3

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


look_back=1
# create and fit the LSTM network
#model = Sequential()
model = Sequential()
model.add(LSTM(16,input_shape=(trainX.shape[1],1),return_sequences=True))
#model.add(LSTM(16,batch_input_shape=(10,1),return_sequences=True))
#model.add(LSTM(16,input_shape=(None, None),return_sequences=True,stateful=True))

#model.add(LSTM(4,return_sequences=True))
#model.add(LSTM(64,  return_sequences=True))
#model.add(LSTM(32,  return_sequences=True))
#model.add(LSTM(16,  return_sequences=True))
#model.add(LSTM(8,  return_sequences=True))
#model.add(LSTM(4,  return_sequences=True))

#model.add(Dropout(.25))
#model.add(LSTM(16))
# #model.add(Dense(1, activation = 'relu'))
# #model.add(Dense(1, activation='sigmoid'))
#.
model.add(Flatten())
# #model.add(Dense(64))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('relu'))
#model.add(Activation('relu'))
model.compile(loss='mae', optimizer=opt,metrics=["accuracy"])
#model.fit(trainX, trainY,epochs=100, batch_size=1, verbose=1)
model.fit(trainX, trainY,epochs=100, batch_size=1, verbose=1)
# make predictions
score=model.evaluate(X,Y,verbose=1)
print(score[1])

# invert predictions
#scaler = StandardScaler()
'''
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

'''
# plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))
'''
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''