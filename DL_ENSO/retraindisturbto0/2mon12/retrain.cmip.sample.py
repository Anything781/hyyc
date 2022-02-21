#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ['LD_LIBRARY_PATH']='/usr/local/cuda-10.1/lib64'
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
from netCDF4 import Dataset

'''target_mon, lead_mon, convfilter, hiddfilter, chlist, lmont(leadmontarget)
multiple'''

lead_mon = 2
target_mon = 12
tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)
batch_size = 400    #BBB
num_convf = convfilter  # M 30 50
num_hiddf = hiddfilter  # N 30 50
xdim = 72
ydim = 24
zdim = 6
xdim2 = int(xdim/4)
ydim2 = int(ydim/4)
sample_size = 2961
tot_size = 2961
conv_drop = 1.0     #CDRP
hidd_drop = 1.0     #HDRP

'''beishu = 0.4 #按最大梯度的倍数保留数据区域'''
beishu = multiple * 0.1  #乘0.1还是0.001取决于.csh文件中bei步长是多少

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=1))

def init_bias(shape):
    return tf.Variable(tf.random_uniform(shape, minval=-0.01, maxval=0.01))


# Read Data (NetCDF4)
inp1 = Dataset('/mnt/input/CMIP5.input.36mon.1861_2001.nc','r')
inp2 = Dataset('/mnt/input/CMIP5.label.12mon.1863_2003.nc','r')

inpv1 = np.zeros((sample_size,6,24,72))
inpv1[:,0:3,:,:] = inp1.variables['sst1'][0:sample_size,ld_mn1:ld_mn2,:,:]
inpv1[:,3:6,:,:] = inp1.variables['t300'][0:sample_size,ld_mn1:ld_mn2,:,:]

inpv2 = inp2.variables['pr'][0:sample_size,tg_mn,0]

#修改数据               1mon12
m=np.fromfile('/mnt/retraindisturbto0/2mon12/4x4all6to072x24.gdat',dtype=np.float32)
m = m.reshape(72,24)
m1 = np.swapaxes(m,0,1) #交换0, 1维，（72,24）——>（24,72）符合原数据类型
condition = m1.max() * beishu #保留数据的条件
aa = m1 > condition

for mm in range(2961):
    for nn in range(6):
        inpv1[mm,nn,:,:] *= aa
    inpv1[mm] = np.where(inpv1[mm]==-0,0,inpv1[mm])#把-0换为0

reinp1 = np.swapaxes(inpv1,1,3)  # (tdim,zdim,ydim,xdim) -> (tdim,xdim,ydim,zdim)

trX = reinp1[:,:,:,:]
trY = inpv2[:,:]

#with tf.device('/gpu:0'): # Using GPU

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

#tf.squared_difference返回平方差（有多个），tf.reduce_mean计算均值
cost = tf.reduce_mean(tf.squared_difference(py_x, Y))
batch = tf.Variable(0, dtype=tf.float32)
train_op = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(cost)
predict_op = py_x

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Launch the graph in a session
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()

    print('-------------------------------------------------------------------------------')
    print('')
    print('Case: case')
    print('Parameter: opfname')
    print('Ensemblemember')
    print('')
    print('Training...')
    print('')

    for i in range(700):  #epoch
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size,
			                       len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], 
		     p_keep_conv: conv_drop, p_keep_hidden: hidd_drop})
        if i%50 == 0:
          print('Epoch', i+1, ', Cost:',sess.run(cost,feed_dict={X: trX,
                 Y: trY, p_keep_conv: 1, p_keep_hidden: 1}))

    save_path = saver.save(sess,'/mnt/document/lmont/cmip/chlist/0.'+str(multiple)+'/model.ckpt')
                                #mnt/retrain0.2.0.7/1mon1/cmip/c30h30/0.1
    print('lmont,chlist,0.multiple')
    print(condition)

