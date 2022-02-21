#!/usr/bin/env python
import numpy as np

CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
tmp = np.zeros((4,72,24))

for i in range(4):
  a0=np.fromfile('/mnt/cnngradstd/lmont/'+CH_list[i]+'gradstdmean.gdat',dtype=np.float32)
  a0 = a0.reshape(72,24)
  tmp[i] = a0
tmp1 = np.mean(tmp,axis=0)

print(tmp1.max(),tmp1.min())
print(tmp1.shape)
tmp1.astype('float32').tofile('/mnt/cnngradstd/lmont/lmontgradstdCHmean.gdat')
