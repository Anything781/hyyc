#!/usr/bin/env python
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']='0'#设置使用GPU
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import math
from netCDF4 import Dataset
from tempfile import TemporaryFile



target_mon = 12
lead_mon = 1
tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)

num_convf = convfilter
num_hiddf = hiddfilter

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
with tf.Session(config=config) as sess:     
    tf.global_variables_initializer().run()
    #初始化模型
    allEN = np.zeros((10, 36, xdim2, ydim2),dtype=np.float32)  # (36,18,6)

    for i in range(10): # EN10

        saver.restore(sess, '/mnt/output/nino34_1month_12_Transfer_20/chlist/EN' + str(i + 1) + '/model.ckpt')
        #tensorflow框架特征图的大小一般是NxHxWxC。对该特征图求梯度，得到的也是NxHxWxC的矩阵，
        # 代表每个特征的权重。再对前三维就平均就得到了每张特征图的权重。
        # 之后根据权重求所有特征图的累加和，再叠加到原图即得到了heatmap

        # 最后一层特征图值 l3b （36,18,6，M/num_convf）
        conv0 = sess.run(l3b, feed_dict={X: ftX,
                                       p_keep_conv: 1.0, p_keep_hidden: 1.0})

        lastfemap = conv0
        #conv0 = tf.expand_dims(conv0, 3)

        #扩张之后的大小，（36,72,24，M/num_convf),已输出查看,sess.run输出为numpy数组，方便执行后续操作,
        #扩张之后的数据通过热图可以看出 72*24 与 18*6在位置上是一致的，扩张正确
        conv1 = sess.run(tf.image.resize_images(conv0, [72,24], method = 'bilinear'))
        print(conv1.shape)
        resize_lastfemap = conv1

        for m in range(num_convf):
            weight = np.zeros((36,num_convf), dtype=np.float32)
            weight_m = np.zeros((36,num_convf), dtype=np.float32)
            ''' #归一化到 【0,1】
            norm_conv1 = np.zeros((36,72,24),dtype=np.float32)
            for mm in range(36):
                a = conv1[mm,:,:,m]
                #norm_conv1[mm] = (a-a.min()) / (a.max() - a.min())   #(36,72,24)
                #因为存在有的特征图全为1或-1的情况，用上述归一化会出现0/0=nan的情况，所以暂改为softmax归一化
                aa = a.reshape(1728)
                aa = sess.run(tf.nn.softmax(aa))
                norm_conv1[mm] = aa.reshape(72,24)
            print('guiyihua min and max:',norm_conv1[1].min(),norm_conv1[1].max())
            #求张量x的最大最小值，不能像np数组一样直接用x.min()、x.max()，tensor没有这样的方法，用tf.reduce_min/max()
            #用np.min/max()也可能会报错，因为可能tf或者np版本不同不能这样转换tensor
            #但是sess.run之后得到的就是numpy类型的数据，就可以以上操作了'''

            inpv2 = inpv1.copy()  # 初始化inpv2, inpv1: (36,6,24,72)
            reinp2 = np.swapaxes(inpv2,1,3)  #inpv2: (36,72,24,6)
            abs_conv1 = abs(conv1)
            print('abs_conv1 min: ',abs_conv1.min())
            for z in range(6):
                reinp2[:,:,:,z] = reinp2[:,:,:,z] * abs_conv1[:,:,:,m]   #不归一化取绝对值然后与原图点乘

            ftX_sCAM = reinp2[:,:,:,:]
            ftX_sCAM = ftX_sCAM.reshape(-1, xdim, ydim, zdim)

            #原score-CAM中baseline默认为全零的图，所以扰动后的得分就是权重,
            '''但是这里回归问题输出本就是有正有负，不一定负的就是不对的，所以此处baseline为原来未扰动的输出
            差值,并取绝对值，不一定扰动后的预测值小了就是不准了，也有可能是更准了，同理预测值是负的更大或者更
            小时候都不一定准确性降低或增大了'''
            result = sess.run(py_x, feed_dict={X: ftX,
                                     p_keep_conv: 1.0, p_keep_hidden: 1.0})
            result_sCAM = sess.run(py_x,feed_dict={X: ftX_sCAM,
                                p_keep_conv: 1.0, p_keep_hidden: 1.0})
            r = result_sCAM - result
            r = r.reshape(36)
            r = abs(r)
            weight_m[:, m] = r

        # 权重再经过softmax归一化
        for s in range(36):
            weight[s,:] = sess.run(tf.nn.softmax(weight_m[s,:]))

        '''求完所有权重，加权求和 + relu也改为保留绝对值'''
        # 最后一层特征图值 l3b （36,18,6，M）
        conv = sess.run(l3b, feed_dict={X: ftX,
                             p_keep_conv: 1.0, p_keep_hidden: 1.0})

        tmp0 = np.zeros((36,18,6,num_convf),dtype=np.float32)
        for n in range(num_convf):
            for y in range(36):
                tmp0[y,:,:,n] = conv[y,:,:,n] * weight[y,n]
        tmp = np.sum(tmp0,axis=3)  #(36,18,6)
        abs_tmp = abs(tmp)
        allEN[i] = abs_tmp         #(10,36,18,6)

    en_mean = np.mean(allEN,axis=0)   #(36,18,6)

    allEN.astype('float32').tofile('/mnt/scorecam/1mon12/chlist/chlistallENscorecam.gdat')
    en_mean.astype('float32').tofile('/mnt/scorecam/1mon12/chlistENmeanscorecam.gdat')
    #保存resize前后不同大小的特征图，观察是否图形一致
    resize_lastfemap.astype('float32').tofile('/mnt/scorecam/1mon12/chlistlastfemap72x24.gdat')
    lastfemap.astype('float32').tofile('/mnt/scorecam/1mon12/chlistlastfemap18x6.gdat')
print('chlist complete')