import numpy as np
np.set_printoptions(suppress=True)
import struct
import math
import time
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
np.set_printoptions(threshold=np.inf)#完整打印numpy数组

from netCDF4 import Dataset


inpv1 = np.zeros((36,6,24,72), dtype=np.float32) #年，z，y维度，x经度
inp1 = Dataset('E:\hyyc\enso\input\input\GODAS.input.36mon.1980_2015.nc','r')#/mnt/enso     /input/GODAS.input.36mon.1980_2015.nc
inpv1[:,0:3,:,:] = inp1.variables['sst'][0:36,16:19,:,:]
#读取变量sst的数据  SEV：TEV 0:36
inpv1[:,3:6,:,:] = inp1.variables['t300'][0:36,16:19,:,:]

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

m2=np.fromfile(r'E:\hyyc\验证\grad_1month12\X1000000\C50H50\allgrad.gdat',dtype=np.float32)
m2 = m2.reshape(36,10,18,6)
#print(m0[15,:,:])

f = Dataset('E:/hyyc/enso/input/input/GODAS.label.12mon.1982_2017.nc', 'r')
obs = f.variables['pr'][:, 11, 0, 0] #36年 12月份




m3=np.fromfile(r'E:\hyyc\单模型\grad_r\en10grad_r.gdat',dtype=np.float32)
m3 = m3.reshape(10,24,72)
k = 7
#print(m3[k])
#print(m3[k].max(),m3[k].min())

a=[[[5,8,0],[8,6,9]],
[[4,0,8],[1,4,6]],
[[2,4,6],[1,4,2]]]
b = np.array(a)

m3=np.fromfile(r'E:\hyyc\单模型\gradall\gradmean1mon12c50h50en10.gdat',dtype=np.float32)
m3 = m3.reshape(72,24)
print(m3)
print(m3[:,:].max())

m1=np.fromfile(r'E:\hyyc\单模型\gradall\grad1001mon12c50h50en10.gdat',dtype=np.float32)
m2=np.fromfile(r'E:\hyyc\单模型\gradall\grad29611mon12c50h50en10.gdat',dtype=np.float32)
m1 =m1.reshape(100,72,24)
m2 = m2.reshape(2961,72,24)
m1 = abs(m1)
m2 = abs(m2)
tmp = np.zeros((2,72,24))
tmp[0,:,:] = np.mean(m1,axis=0)
tmp[1,:,:] = np.mean(m2,axis=0)
grad1 = np.mean(tmp,axis=0)
print(grad1)
print(grad1.max(),grad1.min())
print(np.maximum(grad1,0.1) )


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

