#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ['LD_LIBRARY_PATH']='/usr/local/cuda-10.1/lib64'
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
from netCDF4 import Dataset
from tempfile import TemporaryFile

target_mon = 12
lead_mon = 1
tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)
num_convf = 50   # C50
num_hiddf = 50   # H50

CH_list = 'C50H50'

xdim = 72
ydim = 24
zdim = 6
xdim2 = int(xdim/4)
ydim2 = int(ydim/4)
sample_size = 0    #SEV 0
test_size = 36      #EEV 36
tot_size = 36       #TEV 36

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=1))

def init_bias(shape):
    return tf.Variable(tf.random_uniform(shape, minval=-0.01, maxval=0.01))

# Input Data
inp1 = Dataset('/mnt/input/GODAS.input.36mon.1980_2015.nc','r')

inpv1 = np.zeros((test_size,zdim,ydim,xdim), dtype=np.float32)
inpv1[:,0:3,:,:] = inp1.variables['sst'][sample_size:tot_size,ld_mn1:ld_mn2,:,:] 
inpv1[:,3:6,:,:] = inp1.variables['t300'][sample_size:tot_size,ld_mn1:ld_mn2,:,:]

'''95年 1month12 C50H50 en10
inpv1[15,:,8:11,36:44]=0#0-15s,180-140W
inpv1[15,:,0:11,36:40]=0#0-55s,180-160W
inpv1[15,:,0:11,28:32]=0#0-55s,140E-160E
inpv1[15,:,0:3,48:71]=0#40s-55s,120W-0
inpv1[15,:,12:15,8:28]=0#5N-20N,40E-140E
inpv1[15,:,12:15,52:68]=0#5N-20N,100W-20W
inpv1[15,:,12:19,44:48]=0#5N-40N,140W-120W
inpv1[15,:,16:19,32:40]=0#25N-40N,160E-160W

inpv1[15,:,9:10,37:43]=0#0-15s,180-140W
inpv1[15,:,1:10,37:39]=0#0-55s,180-160W
inpv1[15,:,1:10,29:31]=0#0-55s,140E-160E
inpv1[15,:,1:2,49:70]=0#40s-55s,120W-0
inpv1[15,:,13:14,9:27]=0#5N-20N,40E-140E
inpv1[15,:,13:14,53:67]=0#5N-20N,100W-20W
inpv1[15,:,13:18,45:47]=0#5N-40N,140W-120W
inpv1[15,:,17:18,33:39]=0#25N-40N,160E-160W
'''

m1=np.fromfile('/mnt/heatmapC50H50EN10/gradcamheatmap.gdat',dtype=np.float32)
m1 = m1.reshape(36,18,6)
m1 = np.swapaxes(m1,1,2) #交换1、2维，（36,18,6）——>（36,6,18）符合原数据类型

aa = np.maximum(m1[15],0)   #只保留正值，即正热图值

b = np.zeros((24,72))
for i in range(6):
    for j in range(18):
        b[4*i:4*i+4,j*4:j*4+4] = aa[i,j]#保留热图大于阈值的值，并扩张为24*72尺寸

#此处是只保留了热图覆盖区域数据，其实就是把空白部分设置为0
b1 = b>0 #提取热图值大于0的数据位置
#不能取 = 0，因为上边已经把负值部分设为0了，再取等于0还是保留了所有数据
for ii in range(6):
    inpv1[15,ii,:,:] *= b1
inpv1[15] = np.where(inpv1[15]==-0,0,inpv1[15])#把-0换为0

print(b[4:8,40:48])
print(b1[4:8,40:48])
print(inpv1[15,0,4:8,40:48])

reinp1 = np.swapaxes(inpv1,1,3) # (tdim,zdim,ydim,xdim) -> (tdim,xdim,ydim,zdim)
teX = reinp1[:,:,:,:] 

#with tf.device('/gpu:0'):

# Define Dimension
teX = teX.reshape(-1, xdim, ydim, zdim)
X = tf.placeholder(tf.float32, [None, xdim, ydim, zdim])

w = init_weights([8, 4, zdim, num_convf])
b = init_bias([num_convf])
w2 = init_weights([4, 2, num_convf, num_convf])
b2 = init_bias([num_convf])
w3 = init_weights([4, 2, num_convf, num_convf])
b3 = init_bias([num_convf])
w4 = init_weights([num_convf * xdim2 * ydim2, num_hiddf])
b4 = init_bias([num_hiddf])
w_o = init_weights([num_hiddf, 1])
b_o = init_bias([1])

  # Drop out
p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

  # Model
l1a = tf.tanh(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME') + b)
l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l1 = tf.nn.dropout(l1, p_keep_conv)

l2a = tf.tanh(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2 = tf.nn.dropout(l2, p_keep_conv)

l3a = tf.tanh(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
l3 = tf.reshape(l3a, [-1, w4.get_shape().as_list()[0]])
l3 = tf.nn.dropout(l3, p_keep_conv)

l4 = tf.tanh(tf.matmul(l3, w4) + b4)
l4 = tf.nn.dropout(l4, p_keep_hidden)

py_x = tf.matmul(l4, w_o) + b_o

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Launch the graph in a session
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()

    saver.restore(sess,'/mnt/output/nino34_1month_12_Transfer_20/'+CH_list+'/EN10'+'/model.ckpt')

    result = sess.run(py_x, feed_dict={X: teX, p_keep_conv:1.0,
                      p_keep_hidden:1.0})

    result.astype('float32').tofile('/mnt/heatmapC50H50EN10/'+'gradcam.onlyhpresult.gdat')

print('validation C50H50 complete')












