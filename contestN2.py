from pandas import DataFrame
from pandas import Series
from pandas import concat
import keras
import xlrd
from keras.callbacks import Callback
import os
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import LSTM
from keras.models import load_model
from math import sqrt
import matplotlib.pyplot as plt
import numpy
import keras.backend as K

d = []
ans = []
ans2d = []
for dirname, dirnames, filenames in os.walk('contestdata'):

    for filename in filenames:
        print("讀取檔案 : ",filename)
        if (filename == '.DS_Store'):
            continue
        myWorkbook = xlrd.open_workbook('contestdata/' + filename)
        mySheets = myWorkbook.sheets()
        mySheet = mySheets[0]
        temparray = []
        tempans = []
        check = True
        thisexcelresult = 0
        for row in range(mySheet.nrows):
            for col in range(mySheet.ncols):
                if (str(mySheet.cell(row, col).value).startswith('加工品質量測結果:')):
                    thisexcelresult = mySheet.cell(row, col).value.replace('加工品質量測結果:', '')



        for row in range(mySheet.nrows):
            if(row ==0):
                continue
            target = []
            for col in range(mySheet.ncols):
                target.append(thisexcelresult)
                break

            tempans.append(target)
        for row in range(mySheet.nrows):
            values = []
            result = []
            check = True
            for col in range(mySheet.ncols):
                if (str(mySheet.cell(row, col).value).startswith('加工品質量測結果:')):
                    check = False
                    result.append(str(mySheet.cell(row, col).value).replace('加工品質量測結果:',''))
                    ans2d.append(result)
                values.append(mySheet.cell(row, col).value)
            if check:
                temparray.append(values)
        d.append(temparray)
        ans.append(tempans)

d = numpy.array(d)
d_min = d.min(axis=(0, 1), keepdims=True)
d_max = d.max(axis=(0, 1), keepdims=True)
dscaled = ((d - d_min)/(d_max - d_min))*2-1

print(dscaled.max(axis=(0,1)))
print(dscaled.min(axis=(0,1)))

print(d.shape)
ans = numpy.array(ans).astype('float32')
ans2d = numpy.array(ans2d).astype('float32')

print(ans.shape)
print()
print()
print()
print(dscaled[-10:,:,:])
print(ans)



# ans_min = ans.min()
# ans_max = ans.max()
# ans = (ans - ans_min)/(ans_max - ans_min)
ans_min = ans.min(axis=(0, 1), keepdims=True)
ans_max = ans.max(axis=(0, 1), keepdims=True)
ans = (ans - ans_min)/(ans_max - ans_min)

ans2dor = ans2d

ans2d_min = ans2d.min(axis=(0), keepdims=True)
ans2d_max = ans2d.max(axis=(0), keepdims=True)
ans2d = ((ans2d - ans2d_min)/(ans2d_max - ans2d_min))*2-1
print(ans2d.max(axis=(0)))
print(ans2d.min(axis=(0)))
print(d)
print(ans)
print(ans2d)
# X_train,y_train = d[:-10,:,:],ans[:-10,:]
# X_test,y_test = d[-10:,:,:],ans[-10:,:]
X_train,y_train = dscaled[:-10,:,:],ans2d[:-10,:]
X_test,y_test = dscaled[:,:,:],ans2d[:,:]
print(X_train)
print(X_train.max(axis=(0,1)))
print(X_train.min(axis=(0,1)))
print(y_train)
print(y_train.max(axis=(0)))
print(y_train.min(axis=(0)))

print(X_test)
print(y_test)



def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

class CollectWeightCallback(Callback):
    def __init__(self, layer_index):
        super(CollectWeightCallback, self).__init__()
        self.layer_index = layer_index
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.layers[self.layer_index]
        self.weights.append(layer.get_weights())
        print(layer.get_weights())
cbk = CollectWeightCallback(layer_index=-1)
cbk2 = CollectWeightCallback(layer_index=-2)
cbk3 = CollectWeightCallback(layer_index=-3)
cbk4 = CollectWeightCallback(layer_index=-4)
cbk5 = CollectWeightCallback(layer_index=-5)

model2 = Sequential()

model2.add(LSTM(64,batch_input_shape=(1,7500,4),return_sequences=True,activation='tanh',stateful=True))
model2.add(LSTM(128,batch_input_shape=(1,7500,4),return_sequences=True,activation='tanh',stateful=True))
model2.add(LSTM(256,batch_input_shape=(1,7500,4),return_sequences=True,activation='tanh',stateful=True))
model2.add(LSTM(128,batch_input_shape=(1,7500,4),return_sequences=True,activation='tanh',stateful=True))
model2.add(LSTM(64,batch_input_shape=(1,7500,4),return_sequences=False,activation='tanh',stateful=True))
model2.add(Dense(8,activation='tanh'))
model2.add(Dense(1,activation='tanh'))
adam = keras.optimizers.Adam(lr=0.0001)

model2.compile(loss=root_mean_squared_error, optimizer=adam)
losstraing = []
# lossvalidation = []



history = model2.fit(X_train, y_train, epochs=2, batch_size=1, verbose=1, shuffle=False)

model2.save('firsttryN2')







plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss from contest')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


model = load_model('firsttryN2',custom_objects={ 'root_mean_squared_error':root_mean_squared_error })
model.compile(loss=root_mean_squared_error, optimizer='adam')
predictions = model.predict(X_test,batch_size=1)

print(predictions)
print(y_test)

print('Root mean squared error of prediction all',numpy.sqrt(numpy.mean((predictions[:,:]-y_test[:,:])**2)))
print('Root mean squared error of prediction top30',numpy.sqrt(numpy.mean((predictions[:30,:]-y_test[:30,:])**2)))
print('Root mean squared error of prediction last10',numpy.sqrt(numpy.mean((predictions[-10:,:]-y_test[-10:,:])**2)))


predictions = predictions.ravel()
y_test  = y_test.ravel()

x = numpy.linspace(1,40,40)
print(x)
plt.plot(x,predictions,label='predictions')
plt.plot(x,y_test,label='y_test')
plt.legend()
plt.show()
print('\n\t============================\t\n')
X_test,y_test = d[-10:,:,:],ans2dor[:,:]
print(X_test)
print(y_test)
y_test_r = ((ans2d+1)/2)*(ans2d_max-ans2d_min)+ans2d_min
plt.plot(x,ans2dor,label='ans2dor')
plt.plot(x,ans2d,label='ans2d')
plt.legend()
plt.show()
predictions_r =  ((predictions+1)/2)*(ans2dor.max(axis=(0))-ans2dor.min(axis=(0)))+ans2dor.min(axis=(0))
plt.plot(x,predictions_r,label='predictions_r')

plt.plot(x,y_test_r,label='y_test_r')
plt.legend()
plt.show()


