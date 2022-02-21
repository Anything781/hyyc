#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ['LD_LIBRARY_PATH']='/usr/local/cuda-10.1/lib64'
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
from netCDF4 import Dataset
from tempfile import TemporaryFile
import random


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
    #inpv2 (36,6，24,72）
    inpv2 = np.zeros((test_size, zdim, ydim, xdim), dtype=np.float32)
    #生成一个4 * 4大小的随机数 r


    influence = np.zeros((36, 6, 18), dtype=np.float32)
    for k in range(36):
        saver.restore(sess, '/mnt/output/nino34_1month_12_Transfer_20/' + CH_list + '/EN10' + '/model.ckpt')
        for i in range(0, 24, 4):
            for j in range(0, 72, 4):
                # 要用.copy，否则改变2的时候1也会变，直接等于只是浅复制
                inpv2 = inpv1.copy()  # 初始化inpv2
                # 每次循环生成一组4*4标准正态分布值，共6*18组
                r = np.random.randn(4, 4)

                inpv2[k, :, i:i + 4, j:j + 4] += r

                reinp2 = np.swapaxes(inpv2, 1, 3)  # (tdim,zdim,ydim,xdim) -> (tdim,xdim,ydim,zdim)
                teX_d = reinp2[:, :, :, :]
                teX_d = teX_d.reshape(-1, xdim, ydim, zdim)

                result = sess.run(py_x,feed_dict={X: teX[k, :, :, :].reshape(1, xdim, ydim, zdim), p_keep_conv: 1.0,
                                             p_keep_hidden: 1.0})
                result_d = sess.run(py_x,feed_dict={X: teX_d[k, :, :, :].reshape(1, xdim, ydim, zdim), p_keep_conv: 1.0,
                                               p_keep_hidden: 1.0})
                y = result_d - result
                y = abs(y)      #取绝对值,y当作该像素块的影响值
                aa = int(i / 4)
                bb = int(j / 4)
                influence[k, aa, bb] = y
    print('6x1801 standard normal :')
    print(inpv2[35, 0, 20:24, 68:72])
    influence.astype('float32').tofile('/mnt/disturbinput/addstandardnorm6x1801.gdat')

print('c50h50 complete')
