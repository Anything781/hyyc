import numpy as np
np.set_printoptions(suppress=True)
import struct
import math
import time
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
np.set_printoptions(threshold=np.inf)#完整打印numpy数组

from netCDF4 import Dataset




'''取的16-19月，每一年有三年36个月的数据，所以取95年第15维的'''

'''#设置阈值threshold为3887时省略打印，3888时正常打印，所以文件里边正好有36*18*6=3888个数据
#numpy对数组长度设置了一个阈值，数组长度<=阈值：完整打印；数组长度>阈值：以省略的形式打印；
#这里的np.inf只是为了保证这个阈值足够大，以至于所有长度的数组都能完整打印
f =open(r'E:\hyyc\enso\heatmap0-17-20\nino34_18month_12_Transfer_20\C30H30\heatmap.gdat','rb')
m = f.read()
print(m)

m2= np.fromfile(r'E:\hyyc\enso\output_heatmap_JFM\nino34_1month_1_Transfer_20\C30H30\heatmap.gdat',dtype=np.float32)
f = m2.reshape(36,18,6)[15,:,:]
print(f)
print(max(m2))

each = np.zeros((2,3,2,2),dtype=np.float32)
#print(each)
#each[0,:,:,:] = np.arange(12).reshape(3,2,2)
#把后边的322维度的值赋值给第0维下边所有的322维度的值
each[:,0:2,:,:] = np.arange(8).reshape(2,2,2)
#把后边222维度的值赋值给each两个第一维里边第二维只赋值前两层，
# 切片冒号 ：有数值限制，表示对该维度赋值的限制，

#each[1,:,:,:] = np.arange(12).reshape(3,2,2)
each_mean = np.mean(each,axis=0)
std_each = np.std(each,axis=0)

a = np.zeros((2,3,2,2),dtype=np.float32)
b = np.ones((2,6,2,2),dtype=np.float32)
b[0,:,:,:] = a.reshape(18-12,2,2)[:,:,:]
#a[:,0:1,:,:] = b[:,0:1,:,:]
print(b)
#a = np.sqrt(36)
#print(each.shape[0])

f =np.fromfile(r'E:\hyyc\enso\result\C30H30\result.gdat',dtype=np.float32)
print(f)
f =np.fromfile(r'E:\hyyc\enso\result\C30H50\result.gdat',dtype=np.float32)
print(f)
'''
'''
m= np.fromfile(r'E:\hyyc\enso\oneyear\grad_cam_heatmap_12\nino34_18month_12_Transfer_20\C50H30\heatmap.gdat',dtype=np.float32)
f = m.reshape(36,18,6)[15,:,:]
m2= np.fromfile(r'E:\hyyc\enso\oneyear\grad_cam_heatmap_mean\nino34_18month_12_Transfer_20\C50H50\heatmap.gdat',dtype=np.float32)
f2 = m2.reshape(36,18,6)[15,:,:]


print(f2)
print(f2.max())

#print(np.max(m))
'''
'''
ipth1 = 'E:/hyyc/enso/oneyear/grad_cam_heatmap_1/'
CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']

heat_each = np.zeros((4,36,18,6), dtype=np.float32)
for i in range(4):
  f = open(ipth1+'nino34_18month_12_Transfer_20/'+CH_list[i]+'/heatmap.gdat','r')
  #f文件里有36*18*6个数据，30/50都是这么多
  heat_each[i,:,:,:] = np.fromfile(f, dtype=np.float32).reshape(36,18,6)#[:37,:,:]
                                     #reshape之后进行切片操作再赋值给前边的i
heat_each_mean = np.mean(heat_each, axis=0)
print(heat_each_mean[15,:,:].max())
'''


f = Dataset('E:/hyyc/enso/input/input/GODAS.label.12mon.1982_2017.nc', 'r')
obs = f.variables['pr'][:, 11, 0, 0] #36年 12月份


a=[[[5,8,0],[8,6,9]],
[[4,0,8],[1,4,6]],
[[2,4,6],[1,4,2]]]
b = np.array(a)


'''
#修改grad 删掉陆地上的梯度
grad = np.swapaxes(grad1,0,1)  #(72,24)-->(24,72)符合输入数据

tmp1 = np.zeros((24,72))
for mm in range(24):
    for nn in range(72):
        bb = np.sum(inpv1[14,:,mm,nn])
        if bb == 0:
            bb = 0
        else:
            bb = 1
        tmp1[mm,nn] = bb
print(inpv1[14,:,mm,nn])
grad2 = grad * tmp1
grad3 = np.swapaxes(grad2,0,1) #再转换成习惯的（72,24）
grad3.astype('float32').tofile('E:\hyyc\单模型\gradall\gradmean1mon12c50h50en10del0.gdat')
'''

'''
#按比例 保留数据
grad1=np.fromfile(r'E:\hyyc\单模型\gradall\gradmean1mon12c50h50en10del0.gdat',dtype=np.float32)
grad1 = grad1.reshape(72,24)
grad = np.swapaxes(grad1,0,1)  #(72,24)-->(24,72)符合输入数据
condition = grad.max() * 0.5  #保留海域最大梯度0.5倍的数据
aa = grad > condition    #保留数据的条件
print(aa)
print(condition)
for mm in range(36):
    for nn in range(6):
        inpv1[mm,nn,:,:] *= aa
inpv1= np.where(inpv1==-0,0,inpv1)#把-0换为0
'''
'''  求4个网络结构的平均梯度

a0=np.fromfile(r'E:\hyyc\单模型\grad1mon12\C30H30\gradmeandel0.gdat',dtype=np.float32)
a0 = a0.reshape(72,24)
a1=np.fromfile(r'E:\hyyc\单模型\grad1mon12\C30H50\gradmeandel0.gdat',dtype=np.float32)
a1 = a1.reshape(72,24)
a2=np.fromfile(r'E:\hyyc\单模型\grad1mon12\C50H30\gradmeandel0.gdat',dtype=np.float32)
a2 = a2.reshape(72,24)
a3=np.fromfile(r'E:\hyyc\单模型\grad1mon12\C50H50\gradmeandel0.gdat',dtype=np.float32)
a3 = a3.reshape(72,24)

tmp = np.zeros((4,72,24))
tmp[0] = a0
tmp[1] = a1
tmp[2] = a2
tmp[3] = a3
tmp1 = np.mean(tmp,axis=0)
print(tmp1.max(),tmp1.min())
print(tmp1.shape)
tmp1.astype('float32').tofile('E:\hyyc\单模型\grad1mon12\gradallmeandel0.gdat')
'''
''' 6*18扩张为72*24

ipth1 = 'E:/hyyc/对比实验/disturbinput/all6to0/1mon12/4x4heatmap'

f = open(ipth1+'/'+'4x4all6to072x24.gdat','r')

a = np.fromfile(f, dtype=np.float32).reshape(72,24)

heat_each = a            #第15年 97年  (24,72)  本来就是6x18 行x列 不需要再 np.swapaxes(b[15],0,1)

beishu = 3.5 * 0.1
condition = heat_each.max() * beishu  # 保留数据的条件
aa = heat_each > condition
heat_each1 = heat_each * aa
print(heat_each1.max(), heat_each1.min())
print(beishu)
a = 0
for i in range(72):
    for j in range(24):
        if heat_each1[i,j] > 0:
            a +=1

print(a)
print(72*24)'''


'''整合所有训练数据
ipth1 = 'E:/hyyc/retrain/allretrainresult/'
ipth2 = 'E:/hyyc/retrain0.2-0.7/allretrainresult/'
ipth3 = 'E:/hyyc/retrain0.325-0.675/allretrainresult/'
ipth4 = 'E:/hyyc/retrain0.2-0.7/整合allretrainresult/'
for ii in range(23):
    for j in range(12):
        lead = ii+1
        target = j+1

        f1 = open(ipth1+str(lead)+'mon'+str(target)+'CHmeanresult.gdat','r')
        retrain1= np.fromfile(f1, dtype=np.float32).reshape(9,36)      #retrain0.1-0.9

        f2 = open(ipth2 + str(lead) + 'mon' + str(target) + 'CHmeanresult.gdat', 'r')
        retrain2 = np.fromfile(f2, dtype=np.float32).reshape(11, 36)  # retrain0.2-0.7

        f3 = open(ipth3 + str(lead) + 'mon' + str(target) + 'CHmeanresult.gdat', 'r')
        retrain3 = np.fromfile(f3, dtype=np.float32).reshape(8, 36)   # retrain0.325-0.675

        #0.2-0.7两个模型求平均
        tmp0 = np.zeros((2,6,36))
        for m in range(6):
            tmp0[0,m] = retrain1[m+1]           #retrain0.1 0.2-0.7
            tmp0[1,m] = retrain2[m*2]           #retrain0.2-0.7
        tmp = np.mean(tmp0,axis=0)

        new = np.zeros((22,36))
        new[0] = retrain1[0]    #0.1
        new[1] = tmp[0]         #0.2
        new[2] = retrain2[1]    #0.25
        new[3] = tmp[1]         #0.3
        new[4] = retrain3[0]    #0.325
        new[5] = retrain2[3]    #0.35
        new[6] = retrain3[1]    #0.375
        new[7] = tmp[2]         #0.4
        new[8] = retrain3[2]    #0.425
        new[9] = retrain2[5]    #0.45
        new[10] = retrain3[3]   #0.475
        new[11] = tmp[3]        #0.5
        new[12] = retrain3[4]   #0.525
        new[13] = retrain2[7]   #0.55
        new[14] = retrain3[5]   #0.575
        new[15] = tmp[4]        #0.6
        new[16] = retrain3[6]   #0.625
        new[17] = retrain2[9]   #0.65
        new[18] = retrain3[7]   #0.675
        new[19] = tmp[5]        #0.7
        new[20] = retrain1[7]   #0.8
        new[21] = retrain1[8]   #0.9

        new.astype('float32').tofile(ipth4+str(lead)+'mon'+str(target)+'CHmeanresult.gdat')
        print(str(lead)+'mon'+str(target),'finish')
'''


'''cmip = Dataset('E:/hyyc/enso/input/input/CMIP5.input.36mon.1861_2001.nc','r')
cmiplab = Dataset('E:/hyyc/enso/input/input/CMIP5.label.12mon.1863_2003.nc','r')
soda = Dataset('E:/hyyc/enso/input/input/SODA.input.36mon.1871_1970.nc', 'r')
sodalab = Dataset('E:/hyyc/enso/input/input/SODA.label.12mon.1873_1972.nc', 'r')
godas = Dataset('E:/hyyc/enso/input/input/GODAS.input.36mon.1980_2015.nc', 'r')
godaslab = Dataset('E:/hyyc/enso/input/input/GODAS.label.12mon.1982_2017.nc', 'r')

cmip00 = cmip.variables['sst1'][:,:,:,:]
cmip01 = cmip.variables['t300'][:,:,:,:]

soda00 = soda.variables['sst'][:,:,:,:]
soda01 = soda.variables['t300'][:,:,:,:]

godas00 = godas.variables['sst'][:,:,:,:]
godas01 = godas.variables['t300'][:,:,:,:]

print('cmip sst max:',cmip00.max(),'cmip sst min:',cmip00.min())
print('cmip t300 max:',cmip01.max(),'cmip t300 min:',cmip01.min())
print('soda sst max:',soda00.max(),'soda sst min:',soda00.min())
print('soda t300 max:',soda01.max(),'soda t300 min:',soda01.min())

print('godas sst max:',godas00.max(),'godas sst min:',godas00.min())
print('godas t300 max:',godas01.max(),'godas t300 min:',godas01.min())


cmiplable = cmiplab.variables['pr'][:,:,0]
sodalable = sodalab.variables['pr'][:,:,0]
godaslable = godaslab.variables['pr'][:,:,0]

print('cmip lable max:',cmiplable.max(),'cmip lable min:',cmiplable.min())
print('soda lable max:',sodalable.max(),'soda lable min:',sodalable.min())
print('godas lable max:',godaslable.max(),'godas lable min:',godaslable.min())
sodalable1 = sodalab.variables['pr'][:,:]
print('sodalable1shape:',sodalable)
'''
#!/usr/bin/env python
import numpy as np
from tempfile import TemporaryFile

CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']

a0=np.fromfile('E:/hyyc/retrain/retrainrelu/allrelu/allretrainresult/1mon12CHmeanresult.gdat',dtype=np.float32)
a0 = a0.reshape(36)  #10个



print(a0)
x = np.arange(1,8)
print(x)
