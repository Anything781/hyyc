#!/usr/bin/env python
import numpy as np

tmp = np.zeros((23,12,72,24))

for i in range(23):
  for j in range(12):
    a0=np.fromfile('/mnt/cnngradstd/'+str(i+1)+'mon'+str(j+1)+'/'+str(i+1)+'mon'+str(j+1)+'gradstdCHmean.gdat',dtype=np.float32)
    a0 = a0.reshape(72,24)
    tmp[i,j] = a0

print(tmp.shape)
tmp.astype('float32').tofile('/mnt/cnngradstd/allgradstdcombination.gdat')