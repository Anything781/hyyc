#!/usr/bin/env python
import numpy as np
from tempfile import TemporaryFile

'''1x1 (36,6,24,72)   2x2 (36,6,12,36)   4x4 (36,6,6,18)  c30h30en10'''

CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
tmp1 = np.zeros((4,36,18,6),dtype=np.float32)
tmp2 = np.zeros((4,10,36,18,6),dtype=np.float32)

for i in range(4):
  a1=np.fromfile('/mnt/scorecam/1mon12/'+CH_list[i]+'ENmeanscorecam.gdat',dtype=np.float32)
  a1 = a1.reshape(36,18,6)
  tmp1[i] = a1

tmp1 = np.mean(tmp1,axis=0)
tmp1.astype('float32').tofile('/mnt/scorecam/1mon12/allCHscorecammean.gdat')

print('scorecam mean finish')