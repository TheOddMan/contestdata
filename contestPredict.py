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


def Prepare_Excel_Data():

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

    return d,ans,ans2d

def array_to_numpy(d,ans,ans2d):
    d = numpy.array(d)
    ans = numpy.array(ans).astype("float32")
    ans2d = numpy.array(ans2d).astype("float32")

    return d,ans,ans2d



def zero_mean(value):
    value_zeromean = (value - value.mean(axis=(0, 1))) / value.std(axis=(0, 1))
    return value_zeromean

def negativeOne_to_One_3dim(value):
    value_min = value.min(axis=(0,1),keepdims=True)
    value_max = value.max(axis=(0,1),keepdims=True)
    Scaled_value = ((value-value_min)/(value_max-value_min))*2-1

    return Scaled_value

def negativeOne_to_One_2dim(value):
    value_min = value.min(axis=(0),keepdims=True)
    value_max = value.max(axis=(0),keepdims=True)
    Scaled_value = ((value-value_min)/(value_max-value_min))*2-1

    return Scaled_value
def root_mean_squared_error(y_true, y_pred): #loss函式
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def reverse_NegativeOne_to_One(prediction_value,real_value):
    real_value_min = real_value.min(axis=0)
    real_value_max = real_value.max(axis=0)


    value = (((prediction_value+1)/2)*(real_value_max-real_value_min))+real_value_min
    return value

model = Sequential()
model.add(LSTM(64,batch_input_shape=(1,7500,4),return_sequences=True,stateful=True))
model.add(Dropout(0.5))
model.add(LSTM(128,batch_input_shape=(1,7500,4),return_sequences=True,stateful=True))
model.add(Dropout(0.5))
model.add(LSTM(256,batch_input_shape=(1,7500,4),return_sequences=True,stateful=True))
model.add(Dropout(0.5))
model.add(LSTM(128,batch_input_shape=(1,7500,4),return_sequences=True,stateful=True))
model.add(Dropout(0.5))
model.add(LSTM(64,batch_input_shape=(1,7500,4),return_sequences=False,stateful=True))
model.add(Dropout(0.5))
model.add(Dense(64,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(256,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='tanh'))
adam = keras.optimizers.Adam(lr=0.0001)

model.compile(loss=root_mean_squared_error, optimizer=adam)
model.load_weights('firsttryN1_weight_loss_continue_training')

d,ans,ans2d = Prepare_Excel_Data()

d,ans,ans2d = array_to_numpy(d,ans,ans2d)

Data_Zeromean = zero_mean(d)

Ans2d_NegativeOne_to_One = negativeOne_to_One_2dim(ans2d) #ans2d為原domain答案

X_train,y_train = Data_Zeromean[:30,:,:],Ans2d_NegativeOne_to_One[:30,:] #訓練資料、答案取前30筆 (輸入取原始資料直接轉zscore，答案取介於-1,1)      #答案沒有轉zscore 只有輸入轉zscore
X_test,y_test = Data_Zeromean[:,:,:],Ans2d_NegativeOne_to_One[:,:] #測試資料、答案取全部 (輸入取原始資料直接轉zscore，答案取介於-1,1)      #答案沒有轉zscore 只有輸入轉zscore


# for i in range(301):
#     history = model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=1, shuffle=False)
#     model.reset_states()

import keras.losses
keras.losses.root_mean_squared_error = root_mean_squared_error



modelp = load_model('modelxxx')

modelp.compile(loss=root_mean_squared_error, optimizer='adam')
predictions = modelp.predict(X_test,batch_size=1)

x_axis = numpy.linspace(1,40,40)
predictions = predictions.ravel()
y_test = y_test.ravel()
plt.plot(x_axis,predictions,label='predictions')
plt.plot(x_axis,y_test,label='realvalue')
plt.legend()
plt.show()

RMSE_All_NegativeOne_to_One = numpy.sqrt(numpy.mean((predictions[:]-y_test[:])**2))
RMSE_TOP30_NegativeOne_to_One = numpy.sqrt(numpy.mean((predictions[:30]-y_test[:30])**2))
RMSE_LAST10_NegativeOne_to_One = numpy.sqrt(numpy.mean((predictions[-10:]-y_test[-10:])**2))

print('RMSE_All_NegativeOne_to_One : ',RMSE_All_NegativeOne_to_One)
print('RMSE_TOP30_NegativeOne_to_One : ',RMSE_TOP30_NegativeOne_to_One)
print('RMSE_LAST10_NegativeOne_to_One : ',RMSE_LAST10_NegativeOne_to_One)
print()
ans2d = ans2d.ravel()

predictions_rev = reverse_NegativeOne_to_One(prediction_value=predictions,real_value=ans2d)


plt.plot(x_axis,predictions_rev,label='predictions')
plt.plot(x_axis,ans2d,label='realvalue')
plt.legend()
plt.show()

RMSE_All_Original_Domain = numpy.sqrt(numpy.mean((predictions_rev[:]-ans2d[:])**2))
RMSE_TOP30_Original_Domain = numpy.sqrt(numpy.mean((predictions_rev[:30]-ans2d[:30])**2))
RMSE_LAST10_Original_Domain = numpy.sqrt(numpy.mean((predictions_rev[-10:]-ans2d[-10:])**2))

print('RMSE_All_Original_Domain : ',RMSE_All_Original_Domain)
print('RMSE_TOP30_Original_Domain : ',RMSE_TOP30_Original_Domain)
print('RMSE_LAST10_Original_Domain : ',RMSE_LAST10_Original_Domain)