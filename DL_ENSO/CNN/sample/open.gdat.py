import numpy as np
import struct

np.set_printoptions(threshold=np.inf)
#设置阈值threshold为3887时省略打印，3888时正常打印，所以文件里边正好有36*18*6=3888个数据
#numpy对数组长度设置了一个阈值，数组长度<=阈值：完整打印；数组长度>阈值：以省略的形式打印；
#这里的np.inf只是为了保证这个阈值足够大，以至于所有长度的数组都能完整打印
f =open(r'E:\hyyc\enso\output_heatmap_MJJ _0\nino34_18month_12_Transfer_20\C30H30\heatmap.gdat','rb')
m = f.read()
#print(m)
m = np.fromfile(r'E:\hyyc\enso\output_heatmap_MJJ01\nino34_18month_12_Transfer_20\C30H30\heatmap.gdat',dtype=np.float32)
f = m.reshape(36,18,6)[:,:,:]
print(f)

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
