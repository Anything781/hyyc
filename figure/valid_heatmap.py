#!/usr/bin/env python
from netCDF4 import Dataset
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap, cm, shiftgrid, addcyclic
import numpy as np

deg = u'\xb0'

CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
ipth1 = 'E:/hyyc/valid/cnn'
ipth2 = 'E:/hyyc/valid_NS20'

'''
result = np.zeros((4,36), dtype=np.float32)
for i in range(4):
  f = open(ipth1+'/'+CH_list[i]+'/result_mean.gdat','r')

  result[i,:] = np.fromfile(f, dtype=np.float32)

result_mean = np.mean(result, axis=0)
cnn = result_mean       #real

result_valid = np.zeros((4,36), dtype=np.float32)
for i in range(4):
  f1 = open(ipth2+'/'+CH_list[i]+'/result_mean.gdat','r')

  result_valid[i,:] = np.fromfile(f1, dtype=np.float32)

result_valid_mean = np.mean(result_valid, axis=0)
valid = result_valid_mean       #valid
'''
for i in range(4):
    #for j in range(10):
    m = np.fromfile(ipth1 + '/' + CH_list[i] + '/result_mean.gdat',dtype=np.float32)
    m1 = np.fromfile(ipth2 + '/' + CH_list[i] + '/result_mean.gdat',dtype=np.float32)

    m = m / np.std(m)
    m1 = m1 / np.std(m1)
    # Open observation (GODAS, 1981-2017)
    f = Dataset('E:/hyyc/enso/input/input/GODAS.label.12mon.1982_2017.nc', 'r')
    obs = f.variables['pr'][:, 11, 0, 0] #36年 12月份
    '''
    cnn = cnn / np.std(cnn)       #cnn一维数组,std标准差
    valid = valid / np.std(valid)
    '''
    obs = obs / np.std(obs)


    # Compute correlation coefficient (1984-2017)
    cor_cnn = np.round(np.corrcoef(obs[3:], m[3:])[0, 1], 2)
    #round(x,2)返回浮点数x的四舍五入值,保留2位小数 corrcoef求相关系数

    # Draw Figure
    # Figure 3-(a)
    #plt.subplot(2, 1, 1)
    x = np.arange(2, 38)
    #y = np.arange(4, 38)
    lines = plt.plot( x, m, 'orangered',x, m1, 'dodgerblue',x,obs,'black')#y, sin, 'dodgerblue'
    # x, m, 'orangered',
    my_plot = plt.gca()#plt.gca()用来移动/设置坐标轴
    line0 = my_plot.lines[0]
    line1 = my_plot.lines[1]
    line2 = my_plot.lines[2]
    plt.setp(line0, linewidth=0.5,marker='+',markersize=4)
    plt.setp(line1, linewidth=1, marker='o', markersize=1)
    plt.setp(line2, linewidth=0.5, marker='v', markersize=1)
    plt.legend(('cnn','keep only-SN20°','real'), loc='upper right',
               prop={'size': 7}, ncol=3)
    plt.xlim([0, 38])
    plt.ylim([-3, 3.5])
    plt.xticks(np.arange(2, 39, 2), np.arange(1982, 2019, 2), fontsize=6.5)
    plt.tick_params(labelsize=6., direction='in', length=2, width=0.3, color='black')

    plt.yticks(np.arange(-3, 3.51, 1), fontsize=6.5)
    plt.grid(linewidth=0.2, alpha=0.7)#画网格线
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('(a) 18-month-12 valid', fontsize=8)
    plt.xlabel('Year', fontsize=7)
    plt.ylabel('DJF Nino3.4', fontsize=7)

    plt.savefig(ipth2+'/'+CH_list[i]+'mean.jpg', dpi=1000)
    plt.close()
