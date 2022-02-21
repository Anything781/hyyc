#!/usr/bin/env python
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['CUDA_VISIBLE_DEVICES']='0'#设置使用GPU
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
test_size = 100      #EEV 100
tot_size = 100       #TEV 100年


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
inp1 = Dataset('/mnt/input/SODA.input.36mon.1871_1970.nc','r')

inpv1 = np.zeros((test_size,zdim,ydim,xdim), dtype=np.float32)
inpv1[:,0:3,:,:] = inp1.variables['sst'][sample_size:tot_size,ld_mn1:ld_mn2,:,:]
inpv1[:,3:6,:,:] = inp1.variables['t300'][sample_size:tot_size,ld_mn1:ld_mn2,:,:]


# reshape training data (tdim,zdim,ydim,xdim) -> (tdim,xdim,ydim,zdim)
ftX = np.swapaxes(inpv1,1,3)
#某一点对应的值没变，只是如何看这个多维数组的角度变了

#with tf.device('/gpu:0'):


# Tensors
ftX = ftX.reshape(-1, xdim, ydim, zdim)#第一维根据其他维计算出来
X = tf.placeholder(tf.float32, [None, xdim, ydim, zdim])
# 简单理解下就是占位符的意思，先放在这里，然后在需要的时候给网络传输数据

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
l1a = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME') + b
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

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Launch the graph in a session
with tf.Session(config=config) as sess:     #config=config 根据需要动态申请GPU
    tf.global_variables_initializer().run()
    #初始化模型

    en_mean = np.zeros((100,xdim2,ydim2))#tdim是第一维的个数(36,18,6)

    for k in range(100):
        print('tdim: ' + str(k + 1))

        saver.restore(sess, '/mnt/output/nino34_3month_12_Transfer_20/'+CH_list+'/EN10'+'/model.ckpt')
        # /mnt/  output  /nino34_3month_12_Transfer_20/C50H50/EN...
        conv1 = sess.run(l3b, feed_dict={X: ftX[k, :, :, :].reshape(1, xdim, ydim, zdim),
                                         p_keep_conv: 1.0, p_keep_hidden: 1.0})
        # ftx取出来第一维k层的数据，取出来是该维度下再往下三维的数据，是一个三维的，
        # 再reshap成四维的，  feed_dict用于给之前的占位符赋值，给p_keep_conv赋值为 1.0
        conv2 = sess.run(w4, feed_dict={X: ftX[k, :, :, :].reshape(1, xdim, ydim, zdim),
                                        p_keep_conv: 1.0, p_keep_hidden: 1.0})

        conv3 = sess.run(w_o, feed_dict={X: ftX[k, :, :, :].reshape(1, xdim, ydim, zdim),
                                         p_keep_conv: 1.0, p_keep_hidden: 1.0})

        conv4 = sess.run(b4, feed_dict={X: ftX[k, :, :, :].reshape(1, xdim, ydim, zdim),
                                        p_keep_conv: 1.0, p_keep_hidden: 1.0})

        conv5 = sess.run(b_o, feed_dict={X: ftX[k, :, :, :].reshape(1, xdim, ydim, zdim),
                                         p_keep_conv: 1.0, p_keep_hidden: 1.0})
        # conv1：l3b 卷积激活之后最后一层特征图（1，18，6, M）里边的值            vxyLm
        # conv2：w4  特征图经reshape展开后与全连接层相连的权重 （M*18*6，N）  WxyFmn
        # conv3：w_o 全连接层与输出相连的权重 （N，1）                  Won
        # conv4：b4  连接最后一层卷积层与全连接层的偏置 N                bFn
        # conv5：b_o 连接全连接层和输出的偏置  1

        # conv1:(1, 18, 6, M)

        mul_w = np.zeros((xdim2, ydim2, num_hiddf))  # （18，6，50）

        conv1 = conv1.reshape(xdim2, ydim2, num_convf)  # （18，6，M）
        conv2 = conv2.reshape(xdim2, ydim2, num_convf, num_hiddf)  # （18,6，M，N）
        conv3 = conv3.reshape(num_hiddf)  # N
        conv4 = conv4.reshape(num_hiddf) / (num_convf * xdim2 * ydim2)
        # 比公式bF,n 多除了个num_convf即多除了个 M,那是因为下面代码计算公式for j 循环部分，
        #np.sum 求和2维，axis=2，把+conv4放在了括号里面，所以bFn就多加了M次，要除以M，
        # 或者把+conv4放在np.sum括号之外，就不用多除M了

        # conv5 = conv5.reshape(1)/(xdim2*ydim2*num_hiddf)
        # 比公式bo 多除了个num_hiddf,即多除了个 N
        conv5 = conv5.reshape(1) / (xdim2 * ydim2)

        for j in range(num_hiddf):
            mul_w[:, :, j] = sess.run(
                conv3[j] * tf.tanh(np.sum(conv1[:, :, :] * conv2[:, :, :, j] + conv4[j], axis=2)))
            # Wo*tanh[ M个sum(vxyLm * WxyFmn）+bFn/XlYl ]
        # 后边得到的（18,6）大小的矩阵的值，赋值给mul_w所有第一维j个
        en_mean[k, :, :] = np.sum(mul_w, axis=2) + conv5
           #源码np.mean(mul_w,axis=2)+conv5 公式里是求和,其实求平均和求和都一样，
           # 求平均不过是先求和，再除以该维度的数目而已，值的相对大小关系还是不变
           #sum_f就是热图值hxy  i是代表10个EN

    # save heatmap (4-byte binary)
    en_mean.astype('float32').tofile('/mnt/trainresult3mon12/19683mon12c50h50en10heatmap.gdat')
                      #mnt/output_heatmap_MJJ/nino34_18month_12_Transfer_20/C30H30/heatmap.gdat



