#!/usr/bin/env python
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['CUDA_VISIBLE_DEVICES']='0'#设置使用GPU
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
from netCDF4 import Dataset
from tempfile import TemporaryFile
#from tensorflow.python import pywrap_tensorflow

target_mon = 12
lead_mon = 1
tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn) #23-1+11=33
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
# Variable,它能构建一个变量，主要用来保存tensorflow构建的一些结构中的参数，这样，这些参数才不会随着运算的消失而消失，
# 才能最终得到一个模型。比如神经网络中的权重和bias等，
# 在训练过后，总是希望这些参数能够保存下来，而不是直接就消失了，所以这个时候要用到Variable
# tf.random_normal()函数用于从“服从指定正态分布的序列”中随机取出指定个数的值。
# shape: 输出张量的形状，必选, stddev: 正态分布的标准差，默认为1.0
def init_bias(shape):
    return tf.Variable(tf.random_uniform(shape, minval=-0.01, maxval=0.01))
# shape代表形状，也就是1纬的还是2纬的还是n纬的数组。
# tf.random_uniform((4, 4), minval=low,maxval=high,dtype=tf.float32)))
# 返回4*4的矩阵，产生于low和high之间，产生的值是均匀分布的。

# Input Data
inp1 = Dataset('/mnt/input/GODAS.input.36mon.1980_2015.nc','r')

inpv1 = np.zeros((test_size,zdim,ydim,xdim), dtype=np.float32)
inpv1[:,0:3,:,:] = inp1.variables['sst'][sample_size:tot_size,ld_mn1:ld_mn2,:,:]
inpv1[:,3:6,:,:] = inp1.variables['t300'][sample_size:tot_size,ld_mn1:ld_mn2,:,:]


# reshape training data (tdim,zdim,ydim,xdim) -> (tdim,xdim,ydim,zdim)
ftX = np.swapaxes(inpv1,1,3)  #(tdim,72,24,6)
#某一点对应的值没变，只是如何看这个多维数组的角度变了

#with tf.device('/gpu:0'):


# Tensors
ftX = ftX.reshape(-1, xdim, ydim, zdim)#第一维根据其他维计算出来
X = tf.placeholder(tf.float32, [None, xdim, ydim, zdim])
# 简单理解下就是占位符的意思，先放在这里，然后在需要的时候给网络传输数据

'''定义可训练的输入变量'''
train_input = tf.Variable(tf.random_normal([1,72,24,6], stddev=1),name="input")

w = init_weights([8, 4, zdim, num_convf])#第一层 filter 卷积核的权重
b = init_bias([num_convf])
w2 = init_weights([4, 2, num_convf, num_convf])
b2 = init_bias([num_convf])
w3 = init_weights([4, 2, num_convf, num_convf])
b3 = init_bias([num_convf])
w4 = init_weights([num_convf * xdim2 * ydim2, num_hiddf])
b4 = init_bias([num_hiddf])
w_o = init_weights([num_hiddf, 1])
b_o = init_bias([1])


# Drop out 随机忽略掉一些神经元，减少过拟合
p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

# Model tf.nn.conv2d(input[batch,高度，宽度，通道数],filter[高度，宽度，输入通道数，输出通道数]
# strides:卷积时在图像每一维的步长， 这是一个一维的向量，长度为4
# （分别是[batch方向,height方向,width方向,channels方向）一般batch，channels都是1

'''l1a = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME') + b'''
#输入改为变量train_input
l1a = tf.nn.conv2d(train_input, w, strides=[1, 1, 1, 1], padding='SAME') + b
l1b = tf.tanh(l1a)
l1 = tf.nn.max_pool(l1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l1 = tf.nn.dropout(l1, p_keep_conv)

l2a = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
l2b = tf.tanh(l2a)
l2 = tf.nn.max_pool(l2b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2 = tf.nn.dropout(l2, p_keep_conv)

l3a = tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3
l3b = tf.tanh(l3a)
l3 = tf.reshape(l3b, [-1, w4.get_shape().as_list()[0]])
l3 = tf.nn.dropout(l3, p_keep_conv)

l4 = tf.tanh(tf.matmul(l3, w4) + b4)
l4 = tf.nn.dropout(l4, p_keep_hidden)

py_x = tf.matmul(l4, w_o) + b_o #矩阵相乘
new_py_x = 1000000000 * py_x

#var_list = train_input 只更新变量train_input
train_input_op = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(-new_py_x, var_list=train_input)

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Launch the graph in a session
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    #初始化模型

    # Read data from checkpoint file,
    checkpoint_path = '/mnt/output/nino34_1month_12_Transfer_20/' + CH_list + '/EN10' + '/model.ckpt'
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # Print tensor name and values
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))

    #saver.restore(sess, '/mnt/output/nino34_1month_12_Transfer_20/' + CH_list + '/EN10' + '/model.ckpt')

    #tensorflow框架特征图的大小一般是NxHxWxC。对该特征图求梯度，得到的也是NxHxWxC的矩阵，

    #迭代500次
    for i in range(500):

        tmp0 = sess.run(train_input, feed_dict={p_keep_conv: 1.0, p_keep_hidden: 1.0})
        print('initial input mean:', np.mean(tmp0))
        sess.run(train_input_op, feed_dict={p_keep_conv: 1.0, p_keep_hidden: 1.0})
        result = sess.run(py_x, feed_dict={p_keep_conv: 1.0, p_keep_hidden: 1.0})
        print('output result:', result)
        last_w = sess.run(w4)
        print('last weight mean:', np.mean(last_w))
        all_var = [w, b, w2, b2, w3, b3, w4, b4, w_o, b_o]
        for a in all_var:
            print('var', a, 'mean:', np.mean(sess.run(a)))

        tmp = sess.run(train_input, feed_dict={p_keep_conv: 1.0, p_keep_hidden: 1.0})
        print('train_input mean:', np.mean(tmp))

        # 检查最后一层特征图 l3b 梯度
        grads = tf.gradients(new_py_x, l3b)[0]
        grads = sess.run(grads, feed_dict={p_keep_conv: 1.0, p_keep_hidden: 1.0})
        print('l3b grads mean:', np.mean(grads))

    train_input = sess.run(train_input, feed_dict={p_keep_conv: 1.0, p_keep_hidden: 1.0})

train_input.astype('float32').tofile('/mnt/deepdream/trainrandominput.gdat')

'''
<tf.Variable 'input:0' shape=(1, 72, 24, 6) dtype=float32>, 
<tf.Variable 'Variable:0' shape=(8, 4, 6, 50) dtype=float32>,
<tf.Variable 'Variable_1:0' shape=(50,) dtype=float32>, 
<tf.Variable 'Variable_2:0' shape=(4, 2, 50, 50) dtype=float32>,
<tf.Variable 'Variable_3:0' shape=(50,) dtype=float32>, 
<tf.Variable 'Variable_4:0' shape=(4, 2, 50, 50) dtype=float32>, 
<tf.Variable 'Variable_5:0' shape=(50,) dtype=float32>,
<tf.Variable 'Variable_6:0' shape=(5400, 50) dtype=float32>,
<tf.Variable 'Variable_7:0' shape=(50,) dtype=float32>,
<tf.Variable 'Variable_8:0' shape=(50, 1) dtype=float32>, 
<tf.Variable 'Variable_9:0' shape=(1,) dtype=float32>]

tensor_name:  Variable_9/RMSProp_1
[0.00056939]
tensor_name:  Variable_9
[0.12970252]
last weight mean: 9.708115e-05
initial input mean: 0.0007251707
/usr/local/lib/python3.7/dist-packages/tensorflow/python/client/session.py:1458: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  run_metadata)
output result: [[0.21819273]]
last weight mean: -0.0033452525
'''