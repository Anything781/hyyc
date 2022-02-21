#!/usr/bin/env python
# coding: utf-8

# In[1]:


from netCDF4 import Dataset
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap, cm, shiftgrid, addcyclic
import numpy as np
import cv2

# In[15]:


deg = u'\xb0'
CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']

ipth1 = 'E:/hyyc/enso/output_heatmap_MJJ_sum/'
#ipth2 = 'E:/hyyc/enso/output_heatmap_MJJ'


# In[16]:


# Open Heatmap of each case (1981-2016)  36年
heat_each = np.zeros((4,36,18,6), dtype=np.float32)
for i in range(4):
  f = open(ipth1+'nino34_18month_12_Transfer_20/'+CH_list[i]+'/heatmap.gdat','r')
  #f文件里有36*18*6个数据，30/50都是这么多
  heat_each[i,:,:,:] = np.fromfile(f, dtype=np.float32).reshape(36,18,6)#[:37,:,:]
                                     #reshape之后进行切片操作再赋值给前边的i

heat_each = np.swapaxes(heat_each,2,3)
#把heat_each的第三和第四维度交换，数组内的值也按照同样的方式交换，18*6变成了6*18
#比如原来在(1,30,15,5)位置的值，转换到了(1,30,5,15)。与reshape不同的是
#可以理解为reshape不改变数据的排列顺序，swapaxes改变数据本来的排列顺序

#取平均——————对四个C * H *
heat_each_mean = np.mean(heat_each, axis=0)


# heatmap: 36x6x18 -> 36x6x19,          axis表示沿着/在哪一维添加
ext_heatmap = np.append(heat_each_mean,heat_each_mean[:,:,0:1],axis=2)

# standard deviation 标准差 (36x6x19 -> 6x19)
std_heatmap = np.std(ext_heatmap,axis=0)
#求第0维的标准差，可以看作是求6*19各自对应点36个数的标准差，会得到6*19大小的数组

# mean heatmap (36x6x19 -> 6x19) 36年平均
mean_heatmap = np.mean(ext_heatmap,axis=0)

# significant test 显著性检验
mask_heatmap = np.zeros((36,6,19),dtype=np.float32)
for i in range(36):
  for j in range(6):
    for k in range(19): #abs返回绝对值  sqrt开平方
      #level计算检测统计量公式
      level = abs(ext_heatmap[i,j,k]-mean_heatmap[j,k])/(std_heatmap[j,k]/np.sqrt(36))
      if level > 2.56: # org:2.56; 1.69; 2.03 不同不同
        #反正大概就是当Z>Zα时，为真，Zα为什么取2.56与设定的α有关，此处作者没提
        mask_heatmap[i,j,k] = ext_heatmap[i,j,k]


# In[17]:      可以调用matplotlib中的imshow（）函数来绘制热图

#x，y是尺寸相同的数组，两个数组中同样位置的数据值组成坐标，
# 生成网格点坐标矩阵
x, y  = np.meshgrid(np.arange(0,380,2.5), np.arange(-91.25,91.25,2.5))
# shade1996-1980 mask_heatmap[1996-1981, :, :]
#修改分辨率 标签从82年开始，结果也是从82年开始，所以15就是指的97年的12月不是16
ext_heatmap = np.maximum(ext_heatmap,0)
temp = cv2.resize(ext_heatmap[15,:,:],dsize=(24,76),interpolation=cv2.INTER_LINEAR)

#extent指定热图x和y轴的坐标范围，zorder表示画图先后，数字小的先画
#clim（min，max）设置当前图像的颜色限制
cax = plt.imshow(temp, cmap='RdBu_r',clim=[-2.5,2.5], extent=[0,380,60,-55],zorder=1)

#zorder参数表示画图先后，数字小的先画

plt.gca().invert_yaxis()
#llcrnrlat=左下角纬度,urcrnrlat右上角纬度；llcrnrlon左下角经度, urcrnrlon右上角经度
map = Basemap(projection='cyl', llcrnrlat=-55,urcrnrlat=59, resolution='c',
              llcrnrlon=20, urcrnrlon=380)
map.drawcoastlines(linewidth=0.2)
map.drawparallels(np.arange( -90., 90.,30.),labels=[1,0,0,0],fontsize=6.5,
                  color='grey', linewidth=0.2)
map.drawmeridians(np.arange(0.,380.,60.),labels=[0,0,0,1],fontsize=6.5,
                  color='grey', linewidth=0.2)
map.fillcontinents(color='silver', zorder=2)
space = '                                                               '
plt.title('MJJ 1996 Heatmap'+space+'[97/98 El Niño Case]',fontsize=8, y=0.962,x=0.5)
'''
x = [  120,   280,   280,   120,   120]
y = [  -16,   -16,    23,    23,   -16]
plt.plot(x,y,'black',zorder=4,linewidth=0.9) #画方框，
'''
cax = plt.axes([0.08, 0.28, 0.72, 0.013])#[左，下，宽，高]规定的矩形区域 定义子图
#在已有的 axes 上绘制一个Colorbar，颜色条。
cbar = plt.colorbar(cax=cax, orientation='horizontal')
#对颜色条上参数的设置
cbar.ax.tick_params(labelsize=6.5,direction='out',length=2,width=0.4,color='black')



plt.tight_layout(h_pad=0,w_pad=-0.6)#调整子图减少堆叠
plt.subplots_adjust(bottom=0.10, top=0.9, left=0.08, right=0.8)
plt.savefig(ipth1+'heatmap_all_16.jpg')

