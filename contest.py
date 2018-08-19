from pandas import DataFrame
from pandas import Series
from pandas import concat
import keras
import xlrd
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
dscaled = (d - d_min)/(d_max - d_min)

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
ans2d = (ans2d - ans2d_min)/(ans2d_max - ans2d_min)
print(d)
print(ans)
print(ans2d)
# X_train,y_train = d[:-10,:,:],ans[:-10,:]
# X_test,y_test = d[-10:,:,:],ans[-10:,:]
X_train,y_train = dscaled[:-10,:,:],ans2d[:-10,:]
X_test,y_test = dscaled[:,:,:],ans2d[:,:]
print(X_train)
print()
print(y_train)
print()
print(X_test)
print(y_test)



def root_mean_squared_error(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


model2 = Sequential()
model2.add(LSTM(128,batch_input_shape=(1,7500,4),return_sequences=False,activation='sigmoid',stateful=True))
model2.add(Dense(8,activation='sigmoid'))
model2.add(Dense(1,activation='sigmoid'))
adam = keras.optimizers.Adam(lr=0.000001)

model2.compile(loss=root_mean_squared_error, optimizer=adam)
losstraing = []
# lossvalidation = []

ep = 1001

# for i in range(ep):
#     print('epoch : ',i,'    =================   ')
#     print(
#     )
#     print()
#     history= model2.fit(X_train, y_train, epochs=1, batch_size=1,verbose=1,shuffle=False)
#
#     if i ==0:
#         pass
#     elif i>0:
#      if history.history['loss'] < losstraing[i-1]:
#         print('前一次的loss : ',losstraing[i-1],' 大於這次的loss : ',history.history['loss'],' 儲存model')
#         model2.save('firsttry')
#         model2.save_weights('firstry_weight')
#     else:
#         print('前一次的loss : ', losstraing[i - 1], ' 小於或等於這次的loss : ', history.history['loss'], ' 不儲存model')
#
#     losstraing.append(history.history['loss'])
#     # lossvalidation.append(history.history['val_loss'])
#     model2.reset_states()

# predictions = model2.predict(X_test,batch_size=1)
#
# print('model2 predict : ',predictions)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss from contest')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train','validation'], loc='upper left')
# plt.show()

# losstraing = numpy.array(losstraing).astype('float32')=
# lossvalidation = numpy.array(lossvalidation).astype('float32')

# plt.title('contest plt')=
#
# plt.plot(losstraing)=
# plt.plot(lossvalidation)
# plt.show()=

model = load_model('firsttry',custom_objects={ 'root_mean_squared_error':root_mean_squared_error })
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
plt.plot(x,predictions,label='predictions')
plt.plot(x,y_test,label='y_test')
plt.legend()
plt.show()