from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers import Dense
import numpy as np
import pylab as plt

import numpy as np
import netCDF4
from netCDF4 import Dataset

import tensorflow as tf
seq = tf.keras.models.load_model('./output1/keras_convlstmmodel_ts6_drop_v2_adam_epoch50_combine.h5')

#test data
inp11 = Dataset('GODAS.input.36mon.1980_2015.nc','r')
inp22 = Dataset('GODAS.label.12mon.1982_2017.nc','r')

#time_step = 1 month                                               
sst_11 = np.zeros((35,12,24,72))
t300_11 = np.zeros((35,12,24,72))

sst_11[:,:,:,:] = inp11.variables['sst'][1:,0:12,:,:]
t300_11[:,:,:,:] = inp11.variables['t300'][1:,0:12,:,:]
#(35,12,24,72)

sst_22 = np.zeros((420,24,72))
t300_22 = np.zeros((420,24,72))

for i in range(35):
    sst_22[i*12:(i+1)*12,:,:] = sst_11[i,:,:,:]
    t300_22[i*12:(i+1)*12,:,:] = t300_11[i,:,:,:]
  #(420,24,72)

sst_33 = np.zeros((409,12,24,72))
t300_33 = np.zeros((409,12,24,72))
#滑窗

for i in range(409):
    sst_33[i:,:,:] = sst_22[i:i+12,:,:]
    t300_33[i,:,:,:] = t300_22[i:i+12,:,:]
  #(1177,12,24,72)

#channel = 2
testX = np.zeros((409,12,24,72,2))
testX[:,:,:,:,0] = sst_33
testX[:,:,:,:,1] = t300_33

#label
inpv22 = np.zeros((432))
for i in range(36):
    inpv22[i*12:(i+1)*12] = inp22.variables['pr'][i,:,0,0]
    #(432)

testY = np.zeros((409,24))

#滑窗
for i in range(409):
    testY[i,:] = inpv22[i:i+24]
  #(409,24)

loss,accuracy = seq.evaluate(x = testX,y = testY)
print("----------------------loss/accuracy------------------------")
print(loss,accuracy)

out = seq.predict(testX, batch_size = 5)
cor = np.zeros((24))
for i in range(0,24):
    cor[i] = np.corrcoef(testY[:,i],out[:,i])[0,1]

print("----------------------------cor-----------------------------")
print(cor)

# 绘图
#plt.subplot2grid((2,2),(0,0),rowspan=1,colspan=2) 多子图组合
plt.figure(num=3, figsize=(10, 4))#大小
x = np.arange(0,24,1)
y = np.arange(0,12,1)
lines = plt.plot(x, cor, 'orangered')
my_plot = plt.gca()
line0 = my_plot.lines[0]
plt.setp(line0,linewidth=1.4, marker='o', markersize=2)


plt.legend('CNN',loc='upper right', prop={'size':9}, ncol=5)#右上标
plt.ylabel('Correlation Skill', fontsize=10)
plt.xticks(np.arange(0,24,1), np.arange(1,25,1), fontsize=6) #刻度
#plt.xticks(np.arange(0,24,2)) 
plt.yticks(np.arange(0.3,0.91,0.1), fontsize=9)
plt.ylim([0.25,1.0])
plt.grid(linewidth=0.4, alpha=0.7)#网格
plt.axhline(0.5,color='black',linewidth=0.5)#0.5分界线
plt.title('(a) All-season correlation skills for Nino3.4 (1984-2017)', fontsize=10, x=0.5, y=1)#图表位置
plt.tick_params(labelsize=10,direction='in',length=3,width=0.4,color='black')#刻度大小
plt.savefig('./output1/result.png')
