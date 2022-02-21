#!/usr/bin/env python
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']='0'#设置使用GPU
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
from netCDF4 import Dataset
from tempfile import TemporaryFile
import time

'''target_mon, lead_mon, convfilter, hiddfilter, chlist, lmont(leadmontarget)
'''
tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)
num_convf = convfilter  # M 30 50
num_hiddf = hiddfilter  # N 30 50

xdim = 72
ydim = 24
zdim = 6
xdim2 = int(xdim/4)
ydim2 = int(ydim/4)
sample_size = 0    #SEV 0
test_size = 100      #EEV 36
tot_size = 100       #TEV 36

test_sizecmip = 2961      #EEV 36
tot_sizecmip = 2961       #TEV 36

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
inpv1[:,0:3,:,:] = inp1.variables['sst'][0:tot_size,ld_mn1:ld_mn2,:,:]
inpv1[:,3:6,:,:] = inp1.variables['t300'][0:tot_size,ld_mn1:ld_mn2,:,:]

inpv3 = inpv1.copy()  #复制一个初始输入数据来删掉陆地中的梯度

# Input Data cmip
inp2 = Dataset('/mnt/input/CMIP5.input.36mon.1861_2001.nc','r')

inpv2 = np.zeros((test_sizecmip,zdim,ydim,xdim), dtype=np.float32)
inpv2[:,0:3,:,:] = inp2.variables['sst1'][0:tot_sizecmip,ld_mn1:ld_mn2,:,:]
inpv2[:,3:6,:,:] = inp2.variables['t300'][0:tot_sizecmip,ld_mn1:ld_mn2,:,:]

ftX_cmip = np.swapaxes(inpv2,1,3)
ftX_cmip = ftX_cmip.reshape(-1, xdim, ydim, zdim)#第一维根据其他维计算出来

# reshape training data (tdim,zdim,ydim,xdim) -> (tdim,xdim,ydim,zdim)
ftX = np.swapaxes(inpv1,1,3)
#某一点对应的值没变，只是如何看这个多维数组的角度变了

#with tf.device('/gpu:0'):


# Tensors
ftX = ftX.reshape(-1, xdim, ydim, zdim)#第一维根据其他维计算出来 (100,72,24,6)
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
    grad100 = np.zeros((10, 72, 24, 6))  # test_size是第一维的个数(100,72,24)
    grad2961 = np.zeros((10,72, 24, 6))
    for j in range(10):
        saver.restore(sess, '/mnt/output/nino34_lead_monmonth_target_mon_Transfer_20/chlist/EN' + str(j + 1) + '/model.ckpt')

        '''100年梯度'''
        #tensorflow框架特征图的大小一般是NxHxWxC。对该特征图求梯度，得到的也是NxHxWxC的矩阵，
        # grads_x是列表 （1,100,72,24,6） 加【0】取后四维
        grads_x = tf.gradients(py_x, X)[0]  # (100,72,24,6)
        grads_x = sess.run(grads_x, feed_dict={X: ftX, p_keep_conv: 1.0,
                                               p_keep_hidden: 1.0})
        grads_x = grads_x.reshape(100, 72, 24, 6)
        grads_x = abs(grads_x)                  #取绝对值
        grads_x_mean = np.mean(grads_x, axis=0)  # (72,24,6)

        grad100[j,:,:,:] = grads_x_mean

        '''2961年模拟数据梯度'''
        # grads是列表 （1,2961,72,24,6） 加【0】取后四维
        grads_x_cmip = tf.gradients(py_x, X)[0]  # (1,72,24,6)
        grads_x_cmip = sess.run(grads_x_cmip, feed_dict={X: ftX_cmip, p_keep_conv: 1.0,
                                               p_keep_hidden: 1.0})
        grads_x_cmip = grads_x_cmip.reshape(2961, 72, 24, 6)
        grads_x_cmip = abs(grads_x_cmip)                   #取绝对值
        grads_x_cmip_mean = np.mean(grads_x_cmip, axis=0)  # (72,24,6)

        grad2961[j,:,:,:] = grads_x_cmip_mean   #(10,72,24,6)

    tmp = np.zeros((2,72,24,6))
    tmp[0] = np.mean(grad100,axis=0)
    tmp[1] = np.mean(grad2961,axis=0)
    grad_all_mean = np.mean(tmp,axis=0)  #(72,24,6)

    # 修改grad 删掉陆地上的梯度
    grad = np.swapaxes(grad_all_mean, 0, 1)  # (72,24)-->(24,72)符合输入数据

    tmp1 = np.zeros((24, 72))
    grad2 = np.zeros((24,72,6))
    for mm in range(24):
        for nn in range(72):
            bb = np.sum(inpv3[14, :, mm, nn])
            if bb == 0:
                bb = 0
            else:
                bb = 1
            tmp1[mm, nn] = bb
    for i in range(6):
        grad2[:,:,i]= grad[:,:,i] * tmp1
    grad3 = np.swapaxes(grad2, 0, 1)  # 再转换成习惯的（72,24,6）
    grad3.astype('float32').tofile('/mnt/cnngrad/lmont/chlistgradmeandel0.gdat')
                                  #/mnt/retrain/1mon1/grad/
    grad_all_mean.astype('float32').tofile('/mnt/cnngrad/lmont/chlistgradmean.gdat')
    print('100,2961 min( + 0 or - ):',tmp[0].min(),tmp[1].min())
print('lmont')



