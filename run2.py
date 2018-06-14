#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 21:46:16 2018

@author: sunxinpeng
"""

import tensorflow as tf
import numpy as np
from sklearn import svm


data_num = 60000 #The number of figures
fig_w = 45       #width of each figure


#从文件中读入数据
data = np.fromfile("mnist_train/mnist_train_data",dtype=np.uint8)
label = np.fromfile("mnist_train/mnist_train_label",dtype=np.uint8)

data_test = np.fromfile("mnist_test/mnist_test_data",dtype=np.uint8)
label_test = np.fromfile("mnist_test/mnist_test_label",dtype=np.uint8)

#定义新的feature数组
loss_feature = np.zeros((60000, 10))
loss_feature_test = np.zeros((10000,10))
'''
data = data.reshape(data_num, fig_w*fig_w)
data_test = data_test.reshape(10000, fig_w*fig_w)
data_T = data.T
data_test_T = data_test.T
for i in range(fig_w*fig_w):
    m = np.mean(data_T[i])
    v = data_T.var()
    data_T[i] = (data_T[i] - m)/v
    data_test_T[i] = (data_test_T[i] - m)/v
    
data = data_T.T
data_test = data_test_T.T
'''
#对训练集和测试集的样本分别做normalize
data = data/256
data_test = data_test/256

#将训练集按照label分为10个子集    
data = data.reshape(data_num,fig_w,fig_w)
data_test = data_test.reshape(10000, fig_w, fig_w)
num_0 = np.where(label == 0)[0]
num_1 = np.where(label == 1)[0]
num_2 = np.where(label == 2)[0]
num_3 = np.where(label == 3)[0]
num_4 = np.where(label == 4)[0]
num_5 = np.where(label == 5)[0]
num_6 = np.where(label == 6)[0]
num_7 = np.where(label == 7)[0]
num_8 = np.where(label == 8)[0]
num_9 = np.where(label == 9)[0]

data_0 = data[num_0]
data_1 = data[num_1]
data_2 = data[num_2]
data_3 = data[num_3]
data_4 = data[num_4]
data_5 = data[num_5]
data_6 = data[num_6]
data_7 = data[num_7]
data_8 = data[num_8]
data_9 = data[num_9]
data_split = [data_0,data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9]


tf.reset_default_graph() #图的初始化

batch_size = 64 #每个batch的尺寸

X_in = tf.placeholder(dtype=tf.float32, shape=[None, fig_w, fig_w], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, fig_w, fig_w], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, fig_w * fig_w])#用于计算损失函数
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')#dropout比率

dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels // 2


def lrelu(x, alpha=0.3):#自定义Leaky ReLU函数使效果更好
    return tf.maximum(x, tf.multiply(x, alpha))

#编码
def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, fig_w, fig_w, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)#means
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)#standard deviations            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) #从正态分布中采样
        z  = mn + tf.multiply(epsilon, tf.exp(sd))        
        return z, mn, sd

#解码
def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=fig_w*fig_w, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, fig_w, fig_w])
        return img

#结合
sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

#损失函数
unreshaped = tf.reshape(dec, [-1, fig_w*fig_w])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

for k in range(10):#一共需要训练10个Auto-Encoder
    for i in range(30000):#开始训练
        np.random.shuffle(data_split[k])
        batch = data_split[k][:batch_size]
        sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
        
    
    for i in range(60000):#将训练集样本分别输入训练后的AE得到训练集的loss_feature
        loss_eval = sess.run(loss, feed_dict = {X_in: data[i].reshape(1,45,45), Y: data[i].reshape(1,45,45), keep_prob: 0.8})
        loss_feature[i][k] = loss_eval
        
    for i in range(10000):#将测试集样本分别输入训练后的AE得到训练集的loss_feature
        loss_eval = sess.run(loss, feed_dict = {X_in: data_test[i].reshape(1,45,45), Y: data_test[i].reshape(1,45,45), keep_prob: 0.8})
        loss_feature_test[i][k] = loss_eval
    sess.run(tf.global_variables_initializer())

loss_feature
#SVM分类器的训练与预测  
clf = svm.SVC()
clf.fit(loss_feature, label)
print (clf.score(loss_feature_test, label_test))
    

  
'''
    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
       
        
        
        plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
        plt.show()
        plt.imshow(d[0], cmap='gray')
        plt.show()
        print(i, ls, np.mean(i_ls), np.mean(d_ls))
        


#生成新的字符
randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

for img in imgs:
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img, cmap='gray')
'''
