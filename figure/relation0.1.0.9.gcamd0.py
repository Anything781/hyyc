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

'''CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
'''
'''for a in range(12):
leadmonth = 8
model =  str(leadmonth)+'mon'+str(a+1)
'''

leadmon = 2
targetmon = 12
lmont = str(leadmon)+'mon'+str(targetmon)
ipth1 = 'E:/hyyc/对比实验/gradcam/'+lmont+'/retrain/'                #重训练结果文件夹
ipth2 = 'E:/hyyc/cnnresult/'                             #原cnn预测结果文件夹


#retrain result (9,36) 0.1~0.9 gradient
retrain = open(ipth1 +lmont+'CHmeanresult.gdat', 'r')
retrain = np.fromfile(retrain,dtype=np.float32)
retrain = retrain.reshape(9,36)
'''0.2-0.7是（11,36），从0.2到0.7'''
'''0.1-0.9是（9,36）'''

# Open CNN (1981-2017)
#f = open('/home/jhkim/cnn/output/ver4f_18month_SODA/combination.gdat', 'r')
#cnn = np.fromfile(f, dtype=np.float32)

cnn = open(ipth2 + lmont+'cnnCHmeanresult.gdat', 'r') #原cnn模型全部数据训练预测结果
cnn = np.fromfile(cnn,dtype=np.float32)

# Open observation (GODAS, 1981-2017)
f = Dataset('E:/hyyc/enso/input/input/GODAS.label.12mon.1982_2017.nc', 'r')
mm = targetmon -1
obs = f.variables['pr'][:, mm, 0, 0]
'''36年 a+1 月份'''

for ii in range(9):
    '''retrain有9维，是不同数据训练下的结果，要各自除以各自标准差'''
    retrain[ii] = retrain[ii] / np.std(retrain[ii])

    cnn = cnn / np.std(cnn)       #cnn一维数组,std标准差

    obs = obs / np.std(obs)
    # Compute correlation coefficient (1984-2017)  相关系数
    retrain1 = retrain[ii]
    cor_retrain = np.round(np.corrcoef(obs[3:], retrain1[3:])[0, 1], 2)

    cor_cnn = np.round(np.corrcoef(obs[3:], cnn[3:])[0, 1], 2)
    #round(x,2)返回浮点数x的四舍五入值,保留2位小数 corrcoef求相关系数
    #corrcoef得到的是方形矩阵，大小为两个被求矩阵行数的和，结果为两行数据之间的相关性
    #比如把两个2*2大小的矩阵划分为0-3共4个行，得到的相关性矩阵为2+2=4的4行4列的矩阵
    #如第一行为00,01,02,03行的相关性，第二行为10,11,12,13....
    #所以得到的相关性矩阵是对角线为1，的对称矩阵（因为0和1,1和0的相关性一致）

    # Draw Figure

    #plt.subplot(2, 1, 1)
    x = np.arange(2, 38)
    #x = np.arange(2, 102)
    #y = np.arange(4, 38)
    lines = plt.plot(x, obs, 'black', x, cnn, 'orangered', x, retrain1, 'dodgerblue')#y, sin, 'dodgerblue'
    my_plot = plt.gca()
    line0 = my_plot.lines[0]
    line1 = my_plot.lines[1]
    line2 = my_plot.lines[2]
    plt.setp(line0, linewidth=1.5)
    plt.setp(line1, linewidth=0.5, marker='o', markersize=2)
    plt.setp(line2, linewidth=0.5, marker='v', markersize=2)
    plt.legend(('Observation', 'CNN(Cor=' + str(cor_cnn) + ')','retrain(Cor=' + str(cor_retrain) + ')'),
               loc='upper right', prop={'size': 7}, ncol=3)

    plt.xlim([0, 38])

    #plt.xlim([0, 102])
    plt.ylim([-3, 3.5])
    plt.xticks(np.arange(2, 39, 2), np.arange(1982, 2019, 2), fontsize=6.5)
    #plt.xticks(np.arange(2, 103, 4), np.arange(873, 974, 4), fontsize=6.5)

    plt.tick_params(labelsize=6., direction='in', length=2, width=0.3, color='black')

    plt.yticks(np.arange(-3, 3.51, 1), fontsize=6.5)
    plt.grid(linewidth=0.2, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)

    plt.title(lmont+'gradcam 0.'+str(ii+1)+' retrain result nino3.4 ', fontsize=8)
    plt.xlabel('Year', fontsize=7)
    plt.ylabel('Nino3.4', fontsize=7)

    plt.savefig(ipth1+'retrain0.'+str(ii+1)+'.jpg', dpi=500)
    plt.close()
    #print(model)