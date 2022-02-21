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
lead_mon = 3
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
test_size = 100      #EEV 36
tot_size = 100       #TEV 36

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=1))

def init_bias(shape):
    return tf.Variable(tf.random_uniform(shape, minval=-0.01, maxval=0.01))

# Input Data
inp1 = Dataset('/mnt/input/SODA.input.36mon.1871_1970.nc','r')

inpv1 = np.zeros((test_size,zdim,ydim,xdim), dtype=np.float32)
inpv1[:,0:3,:,:] = inp1.variables['sst'][sample_size:tot_size,ld_mn1:ld_mn2,:,:] 
inpv1[:,3:6,:,:] = inp1.variables['t300'][sample_size:tot_size,ld_mn1:ld_mn2,:,:]

'''66年第95维 3month12 C50H50 en10'''
m1=np.fromfile('/mnt/trainresult3mon12/19683mon12c50h50en10heatmap.gdat',dtype=np.float32)
m1 = m1.reshape(100,18,6)
m1 = np.swapaxes(m1,1,2) #交换1、2维，（36,18,6）——>（36,6,18）符合原数据类型
a = m1[95,:,:]>0  #重要性设置最小阈值,a此时只是一个标签 true false
aa = a*m1[95,:,:]  #此时aa全为正，只保留热图值大于0部分，即删除空白区域数据

b = np.zeros((24,72))
for i in range(6):
    for j in range(18):
        b[4*i:4*i+4,j*4:j*4+4] = aa[i,j]#保留热图大于阈值的值，并扩张为24*72尺寸

#此处是只保留了热图覆盖区域数据，其实就是把空白部分设置为0
b1 = b>0 #提取根据上一步通过阈值保留和丢弃的数据位置，修改阈值时只需要改a那里，此处不需要改
for ii in range(6):
    inpv1[95,ii,:,:] *= b1
inpv1[95] = np.where(inpv1[95]==-0,0,inpv1[95])#把-0换为0

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

    saver.restore(sess,'/mnt/output/nino34_3month_12_Transfer_20/'+CH_list+'/EN10'+'/model.ckpt')

    result = sess.run(py_x, feed_dict={X: teX, p_keep_conv:1.0,
                      p_keep_hidden:1.0})

    result.astype('float32').tofile('/mnt/trainresult3mon12/'+'makeblank0result.gdat')

print('validation C50H50 complete')












