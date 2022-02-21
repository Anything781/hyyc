#!/usr/bin/env python
import numpy as np
from tempfile import TemporaryFile

CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
tmp = np.zeros((4,36))

for i in range(4):
  a0=np.fromfile('/mnt/output/model/'+CH_list[i]+'/ensmean/result.gdat',dtype=np.float32)
  a0 = a0.reshape(36)
  tmp[i] = a0

tmp1 = np.mean(tmp,axis=0)
print(tmp1.max(),tmp1.min())
print(tmp1.shape)
tmp1.astype('float32').tofile('/mnt/cnnresult/lmontcnnCHmeanresult.gdat')