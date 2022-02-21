import numpy as np
np.set_printoptions(suppress=True)
import struct
import math
import time
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
np.set_printoptions(threshold=np.inf)#完整打印numpy数组
from netCDF4 import Dataset

lead_mon = 2
target_mon = 12
tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)
xdim = 72
ydim = 24
zdim = 6
xdim2 = int(xdim/4)
ydim2 = int(ydim/4)
sample_size = 0    #SEV 0
test_size = 100      #EEV 36
tot_size = 100       #TEV 36

'''
CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
'''

inp1 = Dataset('E:\hyyc\enso\input\input\SODA.input.36mon.1871_1970.nc','r')

inpv1 = np.zeros((100,6,24,72), dtype=np.float32)
inpv1[:,0:3,:,:] = inp1.variables['sst'][0:tot_size,ld_mn1:ld_mn2,:,:]
inpv1[:,3:6,:,:] = inp1.variables['t300'][0:tot_size,ld_mn1:ld_mn2,:,:]

inpv3 = inpv1.copy()  #复制一个初始输入数据来删掉陆地中的梯度

m=np.fromfile(r'E:\hyyc\对比实验\gradcam\2mon12\heatmap\allCHgradcammean.gdat',dtype=np.float32)
m = m.reshape(36,18,6)
m1 = np.mean(m,axis=0)  #(18,6)
m1 = np.swapaxes(m1,0,1) #交换1、2维，（18,6）——>（6,18）符合原数据类型
#disturbto0本来就是符合原数据类型（6,18） 不用再np.swapaxes交换维数

gradcam = np.zeros((24,72))
for i in range(6):
    for j in range(18):
        gradcam[4*i:4*i+4,j*4:j*4+4] = m1[i,j]    #扩张为24*72尺寸

#删除陆地区域的热图值
tmp1 = np.zeros((24, 72))
for mm in range(24):
    for nn in range(72):
        bb = np.sum(inpv3[14, :, mm, nn])
        if bb == 0:
            bb = 0
        else:
            bb = 1
        tmp1[mm, nn] = bb
gradcam1 = gradcam * tmp1
gradcam1 = np.swapaxes(gradcam1,0,1)  #再转换成习惯的（72,24）
print(gradcam.shape)
print(gradcam1.shape)
print(gradcam1[4:12,8:12])
print(m1[1:3,2])
gradcam1.astype('float32').tofile('E:/hyyc/对比实验/gradcam/2mon12/heatmap/72x24meangradcam.gdat')

'''m1 = m1.reshape(36,18,6)
m1 = np.swapaxes(m1,1,2) #交换1、2维，（36,18,6）——>（36,6,18）符合原数据类型
a = m1[95,:,:]>0  #重要性设置最小阈值,a此时只是一个标签 true false
aa = a*m1[95,:,:]

b = np.zeros((24,72))
for i in range(6):
    for j in range(18):
        b[4*i:4*i+4,j*4:j*4+4] = aa[i,j]#保留热图大于阈值的值，并扩张为24*72尺寸

b1 = b>0 #提取根据上一步通过阈值保留和丢弃的数据位置
for ii in range(6):
    inpv1[95,ii,:,:] *= b1
inpv1[95] = np.where(inpv1[95]==-0,0,inpv1[95])
print(a[0:2,0:2])
print(b1[0:8,0:8])
print(inpv1[95,1,0:8,0:8])
print(inpv1[95,3,0:8,0:8])'''
