#!/usr/bin/env python
import numpy as np
from tempfile import TemporaryFile

'''1x1 (36,6,24,72)   2x2 (36,6,12,36)   4x4 (36,6,6,18)  c30h30en10'''

CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
tmp1 = np.zeros((4,10,36,6,24,72))
tmp2 = np.zeros((4,10,36,6,12,36))
tmp4 = np.zeros((4,10,36,6,6,18))

'''1x1'''
for i in range(4):
  for j in range(10):
    a1=np.fromfile('/mnt/disturbinput/1month12to0/1x1/0'+CH_list[i]+'EN'+str(j+1)+'disturbto0.gdat',dtype=np.float32)
    a1 = a1.reshape(36,6,24,72)
    tmp1[i,j] = a1

tmp1 = np.mean(tmp1,axis=(0,1))
print('1x1 shape',tmp1.shape)
tmp1.astype('float32').tofile('/mnt/disturbinput/1month12to0/1x1CHmeandisturbto0.gdat')

'''2x2'''
for i in range(4):
  for j in range(10):
    a2=np.fromfile('/mnt/disturbinput/1month12to0/2x2/'+CH_list[i]+'EN'+str(j+1)+'disturbto0.gdat',dtype=np.float32)
    a2 = a2.reshape(36,6,12,36)
    tmp2[i,j] = a2

tmp2 = np.mean(tmp2,axis=(0,1))
print('2x2 shape',tmp2.shape)
tmp2.astype('float32').tofile('/mnt/disturbinput/1month12to0/2x2CHmeandisturbto0.gdat')

'''4x4'''
for i in range(4):
  for j in range(10):
    a4=np.fromfile('/mnt/disturbinput/1month12to0/4x4/'+CH_list[i]+'EN'+str(j+1)+'disturbto0.gdat',dtype=np.float32)
    a4 = a4.reshape(36,6,6,18)
    tmp4[i,j] = a4

tmp4 = np.mean(tmp4,axis=(0,1))
print('4x4 shape',tmp4.shape)
tmp4.astype('float32').tofile('/mnt/disturbinput/1month12to0/4x4CHmeandisturbto0.gdat')

print('disturbto0 mean finish')