import gzip
import os
import sys
from numpy import *
import copy
import time
import pcanet
from sklearn import *
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
fig_w = 45       #width of each figure

def load_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # Load the dataset
    X = np.fromfile("mnist_train/mnist_train_data", dtype=np.uint8)
    y = np.fromfile("mnist_train/mnist_train_label", dtype=np.uint8)
    X = X.reshape(-1, fig_w, fig_w, 1)

    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=.3)

    X_test = np.fromfile("mnist_test/mnist_test_data", dtype=np.uint8)
    y_test = np.fromfile("mnist_test/mnist_test_label", dtype=np.uint8)   

    X_test = X_test.reshape(-1, fig_w, fig_w, 1)

    return [X_train, y_train, X_vali, y_vali, X_test, y_test]


class PCANet(object):
    def __init__(self,NumStages=2,PatchSize=[7,7],NumFilters=[8,8],
                 HistBlockSize=[7,7],BlkOverLapRatio=0.5):
        self.NumStages=NumStages
        self.PatchSize=PatchSize
        self.NumFilters=NumFilters
        self.HistBlockSize=HistBlockSize
        self.BlkOverLapRatio=BlkOverLapRatio

if __name__=='__main__':

    X_train, y_train, X_vali, y_vali, X_test, y_test = load_data()
    
    print('X_train shape:%s'%list(X_train.shape))  # 10000
    print('X_vali shape:%s'%list(X_vali.shape))   # 10000
    print('X_test shape:%s'%list(X_test.shape))    # 50000

    PCANet=PCANet()
    print('\n ====== PCANet Parameters ======= \n')
    print('NumStages= %d'%PCANet.NumStages)
    print('PatchSize=[%d, %d]'%(PCANet.PatchSize[0],PCANet.PatchSize[1]))
    print('NumFilters=[%d, %d]'%(PCANet.NumFilters[0],PCANet.NumFilters[1]))
    print('HistBlockSize=[%d, %d]'%(PCANet.HistBlockSize[0],PCANet.HistBlockSize[1]))
    print('BlkOverLapRatio= %f'%PCANet.BlkOverLapRatio)

    print('\n ====== PCANet Training ======= \n')
    start = time.time()
    ftrain,V,BlkIdx = pcanet.PCANet_train(X_train,PCANet,1)
    end= time.time()
    PCANet_TrnTime=end-start
    print('PCANet training time:%f'%PCANet_TrnTime)

    print('\n ====== Training Linear SVM Classifier ======= \n')
    start = time.time()
    classifier=svm.LinearSVC()
    svm_model= classifier.fit(ftrain,y_train)
    end= time.time()
    LinearSVM_TrnTime=end-start
    print('SVM classfier training time:%f'%LinearSVM_TrnTime)

    print('\n ====== PCANet Testing ======= \n')
    acc=0
    start = time.time()
    for i in range(len(X_test)):
        X_test_i=X_test[i].reshape((1,)+X_test[i].shape)
        ftest,BlkIdx = pcanet.PCANet_FeaExt(PCANet,X_test_i,V)
        pred_label=svm_model.predict(ftest.reshape(1, -1))
        pred_true=y_test[i]
        if pred_label[0]==pred_true:
            acc+=1
            
    end= time.time()
    Averaged_TimeperTest=(end-start)/len(X_test)
    Accuracy=float(acc)/len(X_test)
    ErRate=1-Accuracy
    #print('test accuracy %f'%(float(acc)/len(X_test)))

    print('===== Results of PCANet, followed by a linear SVM classifier =====\n')
    print('     PCANet training time:  %.2f secs.'%PCANet_TrnTime)
    print('     Linear SVM training time:  %.2f secs.'%LinearSVM_TrnTime)
    print('     Testing error rate:  %.2f%%'%(100*ErRate))
    print('     Average testing time  %.2f  secs per test sample. '%Averaged_TimeperTest);
        
