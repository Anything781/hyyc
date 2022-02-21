#!/usr/bin/env python
import numpy as np
from tempfile import TemporaryFile

CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
tmp = np.zeros((4,11,36))

for i in range(4):
  a0=np.fromfile('/mnt/retrain0.2.0.7/lmont/transfer/'+CH_list[i]+'result.gdat',dtype=np.float32)
  a0 = a0.reshape(11,36)  #0.5倍步长算0.2-0.7倍数，共有11个
  tmp[i] = a0

tmp1 = np.mean(tmp,axis=0)
print(tmp1.max(),tmp1.min())
print(tmp1.shape)
tmp1.astype('float32').tofile('/mnt/retrain0.2.0.7/allretrainresult/lmontCHmeanresult.gdat')
print('lmont finish')