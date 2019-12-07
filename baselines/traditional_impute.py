#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:37:18 2018

@author: lyh
"""

from __future__ import print_function
import os
import time
import sys
import argparse
import numpy as np
from sklearn.externals import joblib
sys.path.append("..")

from data import read
from fancyimpute import KNN,MatrixFactorization,IterativeImputer,NuclearNormMinimization

def readdata(missing):
    data_set = read.Read(missing_rate = missing)
    incomplete = []
    complete = []
    m = []
    delta = []
    last_values = []
    for incomplete_t, complete_t, m_t, delta_t, last_values_t in data_set.next(16):
        incomplete.extend(incomplete_t)
        complete.extend(complete_t)
        m.extend(m_t)
        delta.extend(delta_t)
        last_values.extend(last_values_t)

    incomplete = unzip(incomplete)
    complete = unzip(complete)
    m = unzip(m)
    delta = unzip(delta)
    last_values = unzip(last_values)
    return np.array(incomplete), np.array(complete), np.array(m), np.array(delta), np.array(last_values)


def unzip2(data):
    new_data=[]
    for oneclass in data:
        #oneclass:48*31
        newclass=[]
        for one in oneclass:
            newclass.append(one)
        new_data.append(newclass)
    return new_data   
    
    
def unzip(data):
    new_data=[]
    for oneclass in data:
        #oneclass:48*31
        newclass=[]
        for one in oneclass:
            newclass.extend(one)
        new_data.append(newclass)
    return new_data
    
def save(model,path):
    if not os.path.exists(path):
        os.makedirs(path)
    joblib.dump(model, path+"/train_model.m")

def replace0tonan(x,m):
    new_x=[]
    for i in range(len(x)):
        new_x.append([0.0]*len(x[i]))
        for j in range(len(x[i])):
            new_x[i][j]  = float("NaN") if m[i][j] == 0 else x[i][j]
    return new_x
def KNNImpute(X_incomplete,X_complete,M,k,p):
    X_incomplete_new=replace0tonan(X_incomplete,M)
    start_time = time.time()
    X_incomplete_new=np.array(X_incomplete_new)
    X_complete=np.array(X_complete)
    M=np.array(M)
    knn_filled = KNN(k=k).fit_transform(X_incomplete_new)
    mse=np.sum(np.square(knn_filled - X_complete))/np.sum(1-M)
    print(k)
    print(mse)
    print("KNNImpute costed time: %4.4f"%(time.time()-start_time))
    print("\n")
    
    paras=str(p)+"/"+str(k)
    path="knnimpute"+"/"+paras+"/"
    if not os.path.exists(path):
        os.makedirs(path)
    f=open(os.path.join(path,str(mse)),"w")
    f.write(str(mse))
    f.close()
def MICEImpute(X_incomplete,X_complete,M,p):
    start_time = time.time()

    X_incomplete = np.array(replace0tonan(X_incomplete, M))
    X_complete=np.array(X_complete)
    M=np.array(M)
    X_incomplete = IterativeImputer(n_iter=1, n_nearest_features = 15).fit_transform(X_incomplete)
    X_incomplete[np.where(np.isnan(X_incomplete))] = 0.0
    mse=np.sum(np.square(X_incomplete - X_complete))/np.sum(1-M)
    print(mse)
    print("MICEImpute costed time: %4.4f"%(time.time()-start_time))
    print("\n")
    
    paras=str(p)
    path="MICEImpute"+"/"+paras+"/"
    if not os.path.exists(path):
        os.makedirs(path)
    f=open(os.path.join(path,str(mse)),"w")
    f.write(str(mse))
    f.close()
    
    
def MatrixFactorizationImpute(X_incomplete,X_complete,M,p,rank=10,pen=1e-5):
    X_incomplete_new=replace0tonan(X_incomplete,M)
    start_time = time.time()
    X_incomplete_new=np.array(X_incomplete_new)
    X_complete=np.array(X_complete)
    M=np.array(M)
    X_filled_mf = MatrixFactorization(learning_rate=0.01,rank=rank,l2_penalty=pen).fit_transform(X_incomplete_new)
    X_filled_mf[np.where(np.isnan(X_filled_mf))]=0
    mse=np.sum(np.square(X_filled_mf - X_complete))/np.sum(1-M)
    print(str(rank)+"_"+str(pen))
    print(mse)
    print("MatrixFactorizationImpute costed time: %4.4f"%(time.time()-start_time))
    print("\n")
    
    
    paras=str(p)+"/"+str(rank)+"_"+str(pen)
    path="mfimpute"+"/"+paras+"/"
    if not os.path.exists(path):
        os.makedirs(path)
    f=open(os.path.join(path,str(mse)),"w")
    f.write(str(mse))
    f.close()
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--missing-rate', type=float, default=0.5)
    args = parser.parse_args()
    
    incomplete, complete, m, delta, last_values = readdata(args.missing_rate)
    #KNNImpute(incomplete, complete, m, 2, args.missing_rate)
    MICEImpute(incomplete, complete, m, args.missing_rate)
    #MatrixFactorizationImpute(incomplete, complete, m, args.missing_rate, rank=2)
    mean_mse = np.sum(np.square(complete - incomplete))/np.sum(1-m)
    last_mse = np.sum(np.square(complete - last_values))/np.sum(1-m)
    print("mean: %4.7f, last : %4.7f " %(mean_mse, last_mse))



    """
    mse=np.mean(np.square(np.array(x_train) - np.array(x_train_complete)))
    print(mse)
    0.9:0.100166535404
    0.8:0.187361088667
    0.7:0.280390409124
    0.6:0.389332081998
    0.5:0.461659708156
    0.4:0.548353148393
    0.3:0.634718078858
    0.2:0.732961662443
    """
   
    ''' 
    for p in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print(p)
        x_train,m_train,x_train_complete,x_test,m_test,x_test_complete,time_train,time_test,x_last_train=readOrigin(True,p)
        #mse=np.mean(np.square(np.array(x_train) - np.array(x_train_complete)))
        MICEImpute(x_train,x_train_complete,m_train,p)
    '''

    """
    0.9:0.287092074508
    0.8:0.337852819662
    0.7:0.370834079106
    0.6:0.416002661668
    0.5:0.495195285965
    0.4:0.540542871364
    0.3:0.625626119379
    0.2:0.858740877853
    """
    #x_train_gan,x_train_complete_gan,x_train_m_gan,x_test_gan,x_test_complete_gan,x_test_m_gan,time_train_gan,time_test_gan=readImputedData(args.data_path,args.missing_rate)
    #mse=np.mean(np.square(np.multiply(x_train_gan,x_train_m_gan) - np.multiply(x_train_complete_gan,x_train_m_gan)))
    #print(mse)
    
    #KNNImpute(x_train,x_train_complete,m_train,1,0.5)
    #MatrixFactorizationImpute(x_train,x_train_complete,m_train,0.5)
    #calGANImpute(args.data_path,x_train_gan,x_train_complete_gan,args.missing_rate,x_train_m_gan)
   


    
    
