#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ['LD_LIBRARY_PATH']='/usr/local/cuda-10.1/lib64'
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
from netCDF4 import Dataset

'''target_mon, lead_mon, convfilter, hiddfilter, chlist, lmont(leadmontarget)
multiple, document'''

tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)
batch_size = 20    #BBB
num_convf = convfilter  # M 30 50
num_hiddf = hiddfilter  # N 30 50
xdim = 72
ydim = 24
zdim = 7                  #concat之后输入数据就是7个通道了
xdim2 = int(xdim/4)
ydim2 = int(ydim/4)
sample_size = 100
tot_size = 100
conv_drop = 1.0     #CDRP
hidd_drop = 1.0     #HDRP

'''beishu = 0.4 #按最大梯度的倍数保留数据区域'''
#beishu = multiple * 0.1   #乘0.1还是0.001取决于.csh文件中bei步长是多少

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=1))

def init_bias(shape):
    return tf.Variable(tf.random_uniform(shape, minval=-0.01, maxval=0.01))


with tf.device('/cpu:0'):  # Using CPU
    # Read Data (NetCDF4)
    inp1 = Dataset('/mnt/input/SODA.input.36mon.1871_1970.nc', 'r')
    inp2 = Dataset('/mnt/input/SODA.label.12mon.1873_1972.nc', 'r')

    inpv1 = np.zeros((sample_size, 7, 24, 72))
    inpv1[:, 0:3, :, :] = inp1.variables['sst'][0:sample_size, ld_mn1:ld_mn2, :, :]
    inpv1[:, 3:6, :, :] = inp1.variables['t300'][0:sample_size, ld_mn1:ld_mn2, :, :]

    #加上第7维数据，直接加，不作处理
    m = np.fromfile('/mnt/retrain/lmont/grad/lmontgradCHmeandel0.gdat', dtype=np.float32)
    m = m.reshape(72, 24)
    m1 = np.swapaxes(m, 0, 1)  # 交换0, 1维，（72,24）——>（24,72）符合原数据类型

    for mm in range(100):
        inpv1[mm, 6, :, :] = m1

    inpv2 = inp2.variables['pr'][0:sample_size, tg_mn, 0]
    reinp1 = np.swapaxes(inpv1, 1, 3)  # (tdim,6,24,72) -> (tdim,72,24,6)

    trX = reinp1[:, :, :, :]
    trY = inpv2[:, :]

with tf.device('/gpu:0'):
    # Define Dimension
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

    cost = tf.reduce_mean(tf.squared_difference(py_x, Y))
    batch = tf.Variable(0, dtype=tf.float32)
    train_op = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(cost)
    predict_op = py_x

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Launch the graph in a session
with tf.Session(config=config) as sess:
    saver.restore(sess,'/mnt/document/lmont/cmip/chlist/ENnumber/model.ckpt')

    print('-------------------------------------------------------------------------------')
    print('')
    print('Case: case')
    print('Parameter: opfname')#C30H30
    print('Ensemblemember')
    print('')
    print('Training...')
    print('')

    for i in range(20):  #epoch
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size,
			                       len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
		     p_keep_conv: conv_drop, p_keep_hidden: hidd_drop})
        if i%5 == 0:
          print('Epoch', i+1, ', Cost:',sess.run(cost,feed_dict={X: trX,
                 Y: trY, p_keep_conv: 1, p_keep_hidden: 1}))

    save_path = saver.save(sess,'/mnt/document/lmont/transfer/chlist/ENnumber/model.ckpt')
                                #mnt/retrainconcat/1mon1/transfer/c30h30/0.1
    print('lmont,chlist,ENnumber')


