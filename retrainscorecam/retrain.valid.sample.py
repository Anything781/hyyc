#!/usr/bin/env python
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['CUDA_VISIBLE_DEVICES']='0'#设置使用GPU
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
from netCDF4 import Dataset
from tempfile import TemporaryFile

'''target_mon, lead_mon, convfilter, hiddfilter, '''

lead_mon = 1
target_mon = 12
tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)
num_convf = convfilter  # M 30 50
num_hiddf = hiddfilter  # N 30 50

'''CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
CH_list[i]'''

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

inpv1 = np.zeros((36,6,24,72))
inpv1[:,0:3,:,:] = inp1.variables['sst'][sample_size:tot_size,ld_mn1:ld_mn2,:,:] 
inpv1[:,3:6,:,:] = inp1.variables['t300'][sample_size:tot_size,ld_mn1:ld_mn2,:,:]

reinp1 = np.swapaxes(inpv1,1,3) # (tdim,zdim,ydim,xdim) -> (tdim,xdim,ydim,zdim)
teX = reinp1[:,:,:,:] 

#with tf.device('/gpu:number_gpu'):

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
    result = np.zeros((9,36))

    for i in range(9):
        #用i乘0.1的方式读文件夹，可能会出现有多余小数的问题，比如i为2的时候
        #虽然是ii = (i+1)*0.1 = 0.3，但是可能因为精度的问题，ii为0.300000004，就不是想要的0.3了
        #改用组合形式，0.+str（i+1）
        saver.restore(sess,'/mnt/document/lmont/transfer/chlist/0.'+str(i+1)+'/model.ckpt')

        tmp = sess.run(py_x, feed_dict={X: teX, p_keep_conv:1.0,
                          p_keep_hidden:1.0})
        # tmp 数据类型为 （36,1）
        tmp = tmp.reshape(36)
        result[i,:] = tmp
    result.astype('float32').tofile('/mnt/document/lmont/transfer/chlistresult.gdat')

print('validation chlist complete')












