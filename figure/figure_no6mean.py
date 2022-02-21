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
import seaborn as sns #修改颜色条

# In[15]:


deg = u'\xb0'
#CH_list = 'C50H50'

ipth1 = 'E:/hyyc/gradno6mean/1mon1/nodelland'
ipth2 = 'E:/hyyc/retrain/retrainnoleftrighttop/1mon1'



# Open Heatmap of each case (1981-2016)  36年
#heat_each = np.zeros((36,18,6), dtype=np.float32)
f0 = open(ipth1+'/'+'1mon1gradCHmean.gdat','r')  #allCHgradcammean(36,18,6)
heat_each0 = np.fromfile(f0, dtype=np.float32).reshape(72,24,6)

'''tmp0 = np.zeros((72,24,2))
mon = 0
mon1 = mon + 3
tmp0[:,:,0] = heat_each[:,:,mon]
tmp0[:,:,1] = heat_each[:,:,mon1]
heat_each = np.mean(tmp0,axis=2)'''

#for w in range(6):

#heat_each = np.swapaxes(heat_each0[:,:,w], 0, 1)
#把heat_each的第1和第2维度交换，数组内的值也按照同样的方式交换，18*6变成了6*18
#比如原来在(30,15,5)位置的值，转换到了(30,5,15)。与reshape不同的是
#可以理解为reshape不改变数据的排列顺序，swapaxes改变数据本来的排列顺序

heat_each = np.mean(heat_each0,axis=2)
heat_each = np.swapaxes(heat_each,0,1)
for i in range(9):
    beishu = (i+1) * 0.1
    condition = heat_each.max() * beishu  # 保留数据的条件
    aa = heat_each > condition
    heat_each1 = heat_each * aa
    print(heat_each1.max(), heat_each1.min())
    print(beishu)
    # heatmap: 36x6x18 -> 36x6x19,          axis表示沿着/在哪一维添加
    ext_heatmap = np.append(heat_each1,heat_each1[:,0:4],axis=1)

    # standard deviation 标准差 (36x6x19 -> 6x19)
    std_heatmap = np.std(ext_heatmap,axis=0)
    #求第0维的标准差，可以看作是求6*19各自对应点36个数的标准差，会得到6*19大小的数组

    #ext_heatmap = abs(ext_heatmap)
    # mean heatmap (36x6x19 -> 6x19) 36年平均
    mean_heatmap = np.mean(ext_heatmap,axis=0)

    '''# significant test 显著性检验
    mask_heatmap = np.zeros((36,6,19),dtype=np.float32)
    for i in range(36):
      for j in range(6):
        for k in range(19): #abs返回绝对值  sqrt开平方
          #level计算检测统计量公式
          level = abs(ext_heatmap[i,j,k]-mean_heatmap[j,k])/(std_heatmap[j,k]/np.sqrt(36))
          if level > 2.56: # org:2.56; 1.69; 2.03 不同不同
            #反正大概就是当Z>Zα时，为真，Zα为什么取2.56与设定的α有关，此处作者没提
            mask_heatmap[i,j,k] = ext_heatmap[i,j,k]
    '''
    '''
    heatmap = ext_heatmap[15,:,:]
    #heatmap0 = cv2.resize(heatmap,(6,19))
    print(heatmap)
    #uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255。
    #要想将当前的数组作为图像类型来进行各种操作，就要转换到uint8类型
    #方法1 np.uint8(想要转化的变量)
    #这种可能会导致原数据大于255的数被截断，当然此问题不会出现大于255，此处值太小所以乘255
    heatmap = np.uint8(2000*heatmap)
    heatmap = cv2.resize(heatmap,dsize=(640,480),interpolation=cv2.INTER_NEAREST)
    
    #方法2 用cv2.normalize函数配合cv2.NORM_MINMAX，可以设置目标数组的最大值和最小值，
    #然后让原数组等比例的放大或缩小到目标数组，如下面的例子中是将所有数字等比例的放大或缩小到0–255范围的数组中，
    #在不确定数值最大值的时候推荐下面的方法
    
    
    cv2.normalize(heatmap, heatmap, 0, 255, cv2.NORM_MINMAX)
    
    
    heatmap = np.array([heatmap],dtype='uint8')
    heatmap = heatmap.copy()
    print(heatmap)
    
    
    map = cv2.imread('E:\hyyc\enso\Figure_1.png')
    print(map.shape)
    
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    img_heatmap= cv2.addWeighted(map,0.6,heatmap,0.4,0)
    cv2.imshow('title',img_heatmap)#必须要有窗口的名字 title
    cv2.waitKey(0)
    '''
    # In[17]:      可以调用matplotlib中的imshow（）函数来绘制热图

    #x，y是尺寸相同的数组，两个数组中同样位置的数据值组成坐标，
    # 生成网格点坐标矩阵，在此处无用
    #x, y  = np.meshgrid(np.arange(0,380,2.5), np.arange(-91.25,91.25,2.5))
    # shade1996-1980 mask_heatmap[1996-1981, :, :]
    #修改分辨率

    #ext_heatmap = np.maximum(ext_heatmap,0) #去掉小于0

    #temp = cv2.resize(ext_heatmap[15,:,:],dsize=(24,76),interpolation=cv2.INTER_LINEAR)



    #extent指定热图x和y轴的坐标范围，zorder表示画图先后，数字小的先画
    #clim（min，max）设置当前图像的颜色限制
    #标签1873-1972年，此处要看1968年的，应该是在第95
    '''合适的cmap设置，Reds'''

    #修改色集，把某颜色集分成100份，取一部分[20:]
    new_color = sns.color_palette("RdBu_r",100)[20:]
    '''为了增大对比度，可以把clim设置成-0.5max'''

    a = ext_heatmap.max()
    print(a)
    cax = plt.imshow(ext_heatmap, cmap='RdBu_r',clim=[-a,a],
                     interpolation="bicubic", extent=[0,380,60,-55],zorder=3)
    #只通过上边这个把坐标范围限定了之后热图就得到了，后面的cax，subplot之类的只是在调整整个子图的位置
    '''mean_heatmap = np.maximum(mean_heatmap,0)
    cax = plt.imshow(mean_heatmap, cmap='RdBu_r',clim=[-8,8],interpolation="bicubic", extent=[0,380,60,-55],zorder=1)
    print(mean_heatmap.max())
    '''

    #也可加入参数 interpolation="bicubic" 或其他合适插值方法
    #origin='lower/upper'将数组的[0,0]索引放在轴的左上角或左下角。
    plt.gca().invert_yaxis()

    #llcrnrlat=左下角纬度,urcrnrlat右上角纬度；llcrnrlon左下角经度, urcrnrlon右上角经度
    map = Basemap(projection='cyl', llcrnrlat=-55,urcrnrlat=59, resolution='c',
                  llcrnrlon=20, urcrnrlon=380)
    map.drawcoastlines(linewidth=0.2)
    map.drawparallels(np.arange( -90., 90.,30.),labels=[1,0,0,0],fontsize=6.5,
                      color='grey', linewidth=0.2)#画纬线
    map.drawmeridians(np.arange(0.,380.,60.),labels=[0,0,0,1],fontsize=6.5,
                      color='grey', linewidth=0.2)#画经线
    map.fillcontinents(color='silver', zorder=2)

    space = '                                                                      '
    plt.title('1mon1' +' 0.'+str(i+1)+space+'[El Niño Case]',fontsize=8, y=0.962,x=0.5)

    #plt.show()
    cax1 = plt.axes([0.08, 0.28, 0.72, 0.013]) #是为了画颜色条的
    #cax = plt.axes([0.08, 0.28, 0.72, 0.013])#[左，下，宽，高]规定的矩形区域 定义子图 https://www.zhihu.com/question/51745620
    #前两个参数，左，下表示轴域原点坐标
    #在已有的 axes 上绘制一个Colorbar，颜色条。
    cbar = plt.colorbar(cax=cax1, orientation='horizontal')
    #对颜色条上参数的设置
    cbar.ax.tick_params(labelsize=6.5,direction='out',length=2,width=0.4,color='black')

    #plt.tight_layout(h_pad=0,w_pad=-0.6)#调整子图减少堆叠
    plt.subplots_adjust(bottom=0.10, top=0.9, left=0.08, right=0.8)
    plt.savefig(ipth1 +'/0.'+str(i+1)+'.jpg',dpi = 200)#默认dpi=100 dpi=155时突然变方块
    plt.close()


'''
#新增区域ax2,嵌套在ax1内
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25#百分比
# 获得绘制的句柄
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(x,y, 'b')
ax2.set_title('area2')
plt.show()
'''

