#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 21:46:16 2018

@author: sunxinpeng
"""

import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
data_num = 60000 #The number of figures
fig_w = 45       #width of each figure



data = np.fromfile("mnist_train/mnist_train_data",dtype=np.uint8)
label = np.fromfile("mnist_train/mnist_train_label",dtype=np.uint8)

data_test = np.fromfile("mnist_test/mnist_test_data",dtype=np.uint8)
label_test = np.fromfile("mnist_test/mnist_test_label",dtype=np.uint8)

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
data = data/256
data_test = data_test/256
    
data = data.reshape(data_num,-1)
data_test = data_test.reshape(10000, -1)


clf = KNeighborsClassifier(10)
clf.fit(data, label)
print('KNN Classifier')
print (clf.score(data_test, label_test))


clf = svm.SVC()
clf.fit(data, label)
print('svm without dimension reduction')
print (clf.score(data_test, label_test))


  
pca = PCA(10)
pca.fit(data)

data = pca.transform(data)
data_test = pca.transform(data_test)

clf = svm.SVC()
clf.fit(data, label)
print('SVM with dimension reduction')
print (clf.score(data_test, label_test))
    

  
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
