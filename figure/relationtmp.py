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

shape = 'retrainmulthenconcat'
meancormul = np.zeros((23))
meancorcnn = np.zeros((23))

for lead in range(23):

    relationmul = np.zeros((12))
    relationcnn = np.zeros((12))

    for target in range(12):

        model = str(lead+1)+'mon'+str(target+1)
        ipth = 'E:/hyyc/retrain/mul&concat/'+shape+'/relation/'        #保存重训练相关性系数文件夹
        ipth1 = 'E:/hyyc/retrain/mul&concat/'+shape+'/allretrainresult/' #nolfrttop重训练结果文件夹
        ipth2 = 'E:/hyyc/cnnresult/'+model                         #原cnn预测结果文件夹
        #ipth3 = 'E:/hyyc/retrain/allretrainresult/'             #0.1-0.9重训练结果

        #retrain result (7,36) 0.1~0.7 gradient
        retrain = open(ipth1 + model+'CHmeanresult.gdat', 'r')
        retrain = np.fromfile(retrain,dtype=np.float32)
        retrain_mul = retrain.reshape(36)
        '''整合之后的是（7,36）大小'''

        # Open CNN (1981-2017)
        #f = open('/home/jhkim/cnn/output/ver4f_18month_SODA/combination.gdat', 'r')
        #cnn = np.fromfile(f, dtype=np.float32)

        cnn = open(ipth2 + 'cnnCHmeanresult.gdat', 'r') #原cnn模型全部数据训练预测结果
        cnn = np.fromfile(cnn,dtype=np.float32)

        # Open observation (GODAS, 1981-2017)
        f = Dataset('E:/hyyc/enso/input/input/GODAS.label.12mon.1982_2017.nc', 'r')
        obs = f.variables['pr'][:, target, 0, 0]
        '''36年 target月份'''

        retrain_mul = retrain_mul / np.std(retrain_mul)

        cnn = cnn / np.std(cnn)       #cnn一维数组,std标准差

        obs = obs / np.std(obs)
        # Compute correlation coefficient (1984-2017)  相关系数

        cor_mul = np.round(np.corrcoef(obs[3:], retrain_mul[3:])[0, 1], 2)

        cor_cnn = np.round(np.corrcoef(obs[3:], cnn[3:])[0, 1], 2)
        #round(x,2)返回浮点数x的四舍五入值,保留2位小数 corrcoef求相关系数
        #corrcoef得到的是方形矩阵，大小为两个被求矩阵行数的和，结果为两行数据之间的相关性
        #比如把两个2*2大小的矩阵划分为0-3共4个行，得到的相关性矩阵为2+2=4的4行4列的矩阵
        #如第一行为00,01,02,03行的相关性，第二行为10,11,12,13....
        #所以得到的相关性矩阵是对角线为1，的对称矩阵（因为0和1,1和0的相关性一致）


        relationmul[target] = cor_mul
        relationcnn[target] = cor_cnn

    meancorcnn[lead] = np.mean(relationcnn)
    meancormul[lead] = np.mean(relationmul)

sumcorcnn = np.round(np.sum(meancorcnn),2) #np.round 返回四舍五入的 2位小数值
sumcormul = np.round(np.sum(meancormul),2)


#plt.subplot(2, 1, 1)
x = np.arange(1, 24)
#x = np.arange(2, 102)
#y = np.arange(4, 38)
#x = [0.1,0.2,0.25,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65 ,0.675,0.7,0.8,0.9]
lines = plt.plot(x, meancorcnn, 'black',x, meancormul,'orangered')#y, sin, 'dodgerblue'
my_plot = plt.gca()
line0 = my_plot.lines[0]
line1 = my_plot.lines[1]
plt.setp(line0, linewidth=0.5,marker='o', markersize=1.5)
plt.setp(line1, linewidth=0.5, marker='o', markersize=1.5)
plt.legend(('CNN (Sum=' + str(sumcorcnn) +')',shape+' (Sum='+str(sumcormul)+')'),
           loc='lower left', prop={'size': 7}, ncol=3)

'''plt.legend(('Observation', 'CNN(Cor=' + str(cor_cnn) + ')','retrain(Cor=' + str(cor_retrain) + ')'),
           loc='upper right', prop={'size': 7}, ncol=3)'''

plt.xlim([0, 23])#x轴作图范围
plt.ylim([0, 1])           #这两行代码可删，不过画的线位置会有一点点不同，
x_tick = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
#plt.xticks(x,x_tick, fontsize=6.5)#1x,画图范围（间隔）2画图数据
#plt.xticks(np.arange(2, 103, 4), np.arange(873, 974, 4), fontsize=6.5)
plt.xticks(x, x, fontsize=6.5)  #刻度内容

plt.tick_params(labelsize=6., direction='in', length=2, width=0.3, color='black')#刻度线内容

plt.yticks(np.arange(0, 1, 0.1), fontsize=6.5) #刻度内容
plt.grid(linewidth=0.2, alpha=0.7)#显示网格线

#plt.axhline(cor_cnn, color='black', linewidth=0.5)#绘制水平参考线，第一个参数代表画水平线y轴位置

plt.title(shape+' same leadmonth mean Correlation', fontsize=8)
plt.xlabel('lead month', fontsize=7)
plt.ylabel('Correlation', fontsize=7)

#显示每个点数据,plt.text(x坐标，y坐标，数据，其他参数）
'''for a in x:
    plt.text(a,relation[a-1],relation[a-1],horizontalalignment='center', color = 'red',fontsize = 6.5)
 '''
plt.savefig(ipth+shape+'meanCor.jpg', dpi=200)
plt.close()
print(lead)