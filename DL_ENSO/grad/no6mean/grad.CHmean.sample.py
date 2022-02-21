#!/usr/bin/env python
import numpy as np

CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
tmp = np.zeros((4,72,24,6))
tmp2 = np.zeros((4,72,24,6))

for i in range(4):
  a0=np.fromfile('/mnt/cnngrad/lmont/'+CH_list[i]+'gradmeandel0.gdat',dtype=np.float32)
  a0 = a0.reshape(72,24,6)
  tmp[i] = a0
tmp1 = np.mean(tmp,axis=0)

for j in range(4):
  a1=np.fromfile('/mnt/cnngrad/lmont/'+CH_list[j]+'gradmean.gdat',dtype=np.float32)
  a1 = a1.reshape(72,24,6)
  tmp2[j] = a1
tmp3 = np.mean(tmp2,axis=0)

print(tmp1.max(),tmp1.min())
print(tmp1.shape)
tmp1.astype('float32').tofile('/mnt/cnngrad/lmont/lmontgradCHmeandel0.gdat')
tmp3.astype('float32').tofile('/mnt/cnngrad/lmont/lmontgradCHmean.gdat')