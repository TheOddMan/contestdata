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
dscaled_zeromean = (d-d.mean(axis=(0,1)))/d.std(axis=(0,1))
dscaled_zeromean_x = (dscaled-dscaled.mean(axis=(0,1)))/dscaled.std(axis=(0,1))

ans = numpy.array(ans).astype('float32')
ans2d = numpy.array(ans2d).astype('float32')

ans_min = ans.min(axis=(0, 1), keepdims=True)
ans_max = ans.max(axis=(0, 1), keepdims=True)
ans = (ans - ans_min)/(ans_max - ans_min)

ans2dor = ans2d

ans2d_min = ans2d.min(axis=(0), keepdims=True)
ans2d_max = ans2d.max(axis=(0), keepdims=True)
ans2dscaled = ((ans2d - ans2d_min)/(ans2d_max - ans2d_min))*2-1
ans2d_zeromean = (ans2dor - ans2dor.mean(axis=0))/ans2dor.std(axis=0)
ans2d_zeromean_x = (ans2dscaled - ans2dscaled.mean(axis=0))/ans2dscaled.std(axis=0)

dscaled_zeromean_data1 = dscaled_zeromean[0,:,0:1]
d_data1 = d[0,:,0:1]
plt.plot(dscaled_zeromean_data1)
plt.plot(d_data1)
plt.show()
print("原始資料 : ",d.max(axis=(0,1)))                  # [4.19083850e-04            8.54607292e-04               3.32002599e-04             9.34526083e-05]
print(dscaled_zeromean.max(axis=(0,1)))                # [51.38131272                  68.72421881                   61.60185052                 35.46040182]
print(dscaled_zeromean_x.max(axis=(0,1)))


print("原始資料 : ",d.min(axis=(0,1)))                  # [2.80511371e-09           1.31716758e-09                5.27894631e-09           2.50920048e-09]
print(dscaled_zeromean.min(axis=(0,1)))                # [-0.37539297                   -0.3017667                      -0.55067907                 -0.73846001]
print(dscaled_zeromean_x.min(axis=(0,1)))

plt.plot(ans2d_zeromean)
plt.show()
print("原始資料 : ",ans2d.max(axis=0)) # [1.1181]
print(ans2d_zeromean.max(axis=(0))) # [2.6773424]
print(ans2d_zeromean_x.max(axis=(0)))

print("原始資料 : ",ans2d.min(axis=0)) # [0.306]
print(ans2d_zeromean.min(axis=(0))) # [-1.429179]
print(ans2d_zeromean_x.min(axis=(0)))