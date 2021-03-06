#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from netCDF4 import Dataset

#三年为一个整体，前两年24个月是提前期，
tg_mn = int(target_mon - 1)  # taget_mon --TMON 
ld_mn1 = int(23 - lead_mon + tg_mn)  # lead_mon --LEAD
ld_mn2 = int(23 - lead_mon + tg_mn + 3)
batch_size = batsiz
num_convf = convfilter  # conf（30 50） 卷积filter/feature个数(同数量)
num_hiddf = hiddfilter  # hidf（30 50）Number of hidden neurons 隐藏神经元数量
xdim = xxx
ydim = yyy
zdim = zzz
xdim2 = int(xdim / 4)  # 72/4 =18
ydim2 = int(ydim / 4)  # 24/4
sample_size = SAMSIZ  # sss 100 Training data size of training set
tot_size = TOTSIZ  # TTT 10 Total data size of training set
conv_drop = CDRP  # CDD 1.0 drop rate at the convolutional layer
hidd_drop = HDRP  # Hdd 1.0 drop rate at the hidden layer


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

with tf.device('/cpu:0'):  # Using CPU
    # Read Data (NetCDF4)
    inp1 = Dataset('/home/jhkim/data/SODA/samfile', 'r')
    # SMF soda.sst_t300.map.1871_1970.36mon.nc Sample of training data
    inp2 = Dataset('/home/jhkim/data/SODA/labfile', 'r')
    # LBF soda.nino.1873_1972.LAG.nc Label标签 of training data

    inpv1 = np.zeros((sample_size, zzz, yyy, xxx))
    # 返回来一个给定形状和类型的用0填充的数组；这个是四维数组
    inpv1[:, 0:3, :, :] = inp1.variables['sst'][0:sample_size, ld_mn1:ld_mn2, :, :]
    # 第二维切片取0、1、2
    inpv1[:, 3:6, :, :] = inp1.variables['t300'][0:sample_size, ld_mn1:ld_mn2, :, :]

    inpv2 = inp2.variables['pr'][0:sample_size, tg_mn, 0]
    reinp1 = np.swapaxes(inpv1, 1, 3)  # (tdim,zdim,ydim,xdim) -> (tdim,xdim,ydim,zdim)

    trX = reinp1[:, :, :, :]
    trY = inpv2[:, :]

with tf.device('/gpu:number_gpu'):
    # Define Dimension 维
    trX = trX.reshape(-1, xdim, ydim, zdim)

    X = tf.placeholder(tf.float32, [None, xdim, ydim, zdim])
    Y = tf.placeholder(tf.float32, [None, 1])

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
    # 简单理解下就是占位符的意思，先放在这里，然后在需要的时候给网络传输数据
    p_keep_hidden = tf.placeholder(tf.float32)

    # Model
    #tf.nn.conv2d(input[batch,高度，宽度，通道数],filter[高度，宽度，输入通道数，输出通道数]
    # strides:卷积时在图像每一维的步长， 这是一个一维的向量，长度为4
    # （分别是[batch方向,height方向,width方向,channels方向）一般batch，channels都是1

    l1a = tf.tanh(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME') + b)
    # tanh双曲正切曲线函数
    # tf.nn.conv2d (input, filter, strides, padding, use_ cudnn on_ gpu=None,name=None)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #tf.nn.max_pool(value, ksize, strides, padding, data_format, name)
    #value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape。
    #ksize：池化窗口的大小，取一个四维向量，一般是[1, in_height, in_width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1。
    #strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.tanh(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.tanh(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
    l3 = tf.reshape(l3a, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.tanh(tf.matmul(l3, w4) + b4)
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    py_x = tf.matmul(l4, w_o) + b_o#矩阵相乘

    cost = tf.reduce_mean(tf.squared_difference(py_x, Y))
    batch = tf.Variable(0, dtype=tf.float32)
    train_op = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(cost)
    predict_op = py_x

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Launch the graph in a session
with tf.Session(config=config) as sess:
    saver.restore(sess, 'home_directory/output/nino34_lead_monmonth_target_mon/opfname/ENmember/model.ckpt')

    print('-------------------------------------------------------------------------------')
    print('')
    print('Case: case')
    print('Parameter: opfname')
    print('Ensemblemember')
    print('')
    print('Training...')
    print('')

    for i in range(epoch):
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size,
                                                                   len(trX) + 1, batch_size))
        #len（trx）返回数组第一维的长度
        for start, end in training_batch:
            #start：end （0:20），···· （80：100）
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: conv_drop, p_keep_hidden: hidd_drop})
            # 给p_keep_conv赋值为 conv_drop

    save_path = saver.save(sess, 'home_directory/output/case/opfname/ENmember/model.ckpt')


