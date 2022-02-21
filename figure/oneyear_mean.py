import  numpy as np
np.set_printoptions(suppress=True)
'''
np.set_printoptions(precision=3, suppress=True)
参数：
precision: 保留几位小数，后面不会补0
supress: 对很大/小的数不使用科学计数法 (true)
'''

ipth1 = 'E:/hyyc/enso/oneyear/'
CH_list = 'C50H50'

heat_each = np.zeros((12,36,18,6),dtype=np.float32)

for i in range(12):

    f = open(ipth1 + 'grad_cam_heatmap_'+ str(i+1) +'/nino34_18month_12_Transfer_20/'+ CH_list +'/heatmap.gdat', 'r')
    print(ipth1 + 'grad_cam_heatmap_'+ str(i+1) +'/nino34_18month_12_Transfer_20/'+ CH_list +'/heatmap.gdat')
    heat_each[i, :, :, :] = np.fromfile(f, dtype=np.float32).reshape(36, 18, 6)
    print(i)
      #f文件里有36*18*6个数据，30/50都是这么多

for i in  range(12):
    for j in range(36):
        #heat_each +=0.0000001#有一个文件全0防止后边除报错

        #heat_each[i,j,:,:] /=heat_each[i,j,:,:].max()
        print(heat_each[i,j,:,:].max())


heat_each = np.mean(heat_each,axis=0)
print(heat_each.shape)
heat_each.astype('float32').tofile('E:/hyyc/enso/oneyear/grad_cam_heatmap_mean/nino34_18month_12_Transfer_20/'+ CH_list +'/heatmap.gdat')
#此处路径如果是 \ 会报错

'''

import matplotlib.pyplot as plt
import numpy as np
import cv2

np.random.seed(1)

data = np.random.rand(114).reshape(6, 19)
data = cv2.resize(data,dsize=(24,76),interpolation=cv2.INTER_LINEAR)
print(data)
plt.imshow(data)
plt.show()
'''