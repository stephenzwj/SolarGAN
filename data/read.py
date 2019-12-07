# -*- coding: utf-8 -*-  

'''
读数据
ZONEID,TIMESTAMP,VAR78,VAR79,VAR134,VAR157,VAR164,VAR165,VAR166,VAR167,VAR169,VAR175,VAR178,VAR228,POWER
除了id和时间戳之外的属性标准化 √
按照时间间隔划分 √
随机生成mask  √
生成delta √
再shuffle(complete, incplete, delta, m, lastvalues)
返回incomplete和complete


'''

import os
import random
import math
import numpy as np
from sklearn import preprocessing

def sample_M(m, n,k, p):
    #p是missing rate，0.7--> 有70的数据是0

    np.random.seed(10)
    A = np.random.uniform(0., 1., size = [m, n, k])
    B = A > p
    C = 1.*B
    return C

class Read():

    def __init__(self, dataPath1 = "../data/ZONE1.csv",dataPath2 = "../data/ZONE2.csv",dataPath3 = "../data/ZONE3.csv", missing_rate = 0.7, gap=24):
        raw1 = np.loadtxt(dataPath1, dtype=float,delimiter=",",skiprows=(1), usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14))
        raw2 = np.loadtxt(dataPath2, dtype=float,delimiter=",",skiprows=(1), usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14))
        raw3 = np.loadtxt(dataPath3, dtype=float,delimiter=",",skiprows=(1), usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14))
        
        raw = np.concatenate((raw1,raw2,raw3), axis = 1) 
        data = preprocessing.scale(raw)
        # 19704*39
        #data = raw
        i = 0
        data_set = []
        while i + gap <= len(data):
            data_set.append(data[i:i+gap])
            i += gap

        complete_data = np.array(data_set)
        #821*24*13
        m = sample_M(len(complete_data), len(complete_data[0]), len(complete_data[0][0]), missing_rate)
        incomplete_data = m*complete_data

        delta = []
        lastvalues = []
        for h in range(len(m)):
            # oneFile: steps*value_number
            oneFile = incomplete_data[h]
            one_deltaPre = []
            one_lastvalues = []
            one_m = m[h]

            for i in range(len(oneFile)):
                t_deltaPre=[0.0]*len(oneFile[i])
                t_lastvalue=[0.0]*len(oneFile[i])
                one_deltaPre.append(t_deltaPre)
                one_lastvalues.append(t_lastvalue)
                
                if i==0:
                    for j in range(len(oneFile[i])):
                        one_lastvalues[i][j]=0.0 if one_m[i][j]==0 else oneFile[i][j]
                    continue
                for j in range(len(oneFile[i])):
                    if one_m[i-1][j]==1:
                        one_deltaPre[i][j] = 1.0
                    if one_m[i-1][j]==0:
                        one_deltaPre[i][j] = one_deltaPre[i-1][j] + 1.0
                        
                    if one_m[i][j]==1:
                        one_lastvalues[i][j]=oneFile[i][j]
                    if one_m[i][j]==0:
                        one_lastvalues[i][j]=one_lastvalues[i-1][j]


            delta.append(one_deltaPre)
            lastvalues.append(one_lastvalues)
        
        self.incomplete_data = incomplete_data
        self.complete_data = complete_data
        self.m = m
        self.delta = delta
        self.lastvalues = lastvalues
        length = len(self.incomplete_data)

        partition = [0.8,0.9,1.0]

        self.train_incomplete = self.incomplete_data[0:int(length*partition[0])]
        self.train_complete = self.complete_data[0:int(length*partition[0])]
        self.train_m = self.m[0:int(length*partition[0])]
        self.train_delta = self.delta[0:int(length*partition[0])]
        self.train_last = self.lastvalues[0:int(length*partition[0])]



        self.val_incomplete = self.incomplete_data[int(length*partition[0]):int(length*partition[1])]
        self.val_complete = self.complete_data[int(length*partition[0]):int(length*partition[1])]
        self.val_m = self.m[int(length*partition[0]):int(length*partition[1])]
        self.val_delta = self.delta[int(length*partition[0]):int(length*partition[1])]
        self.val_last = self.lastvalues[int(length*partition[0]):int(length*partition[1])]



        self.test_incomplete = self.incomplete_data[int(length*partition[1]):int(length*partition[2])]
        self.test_complete = self.complete_data[int(length*partition[1]):int(length*partition[2])]
        self.test_m = self.m[int(length*partition[1]):int(length*partition[2])]
        self.test_delta = self.delta[int(length*partition[1]):int(length*partition[2])]
        self.test_last = self.lastvalues[int(length*partition[1]):int(length*partition[2])]



    def next_train(self, batch_size):
        c = list(zip(self.train_incomplete, self.train_complete, self.train_m, self.train_delta, self.train_last))
        random.shuffle(c)
        self.train_incomplete, self.train_complete, self.train_m, self.train_delta, self.train_last = zip(*c)

        i = 1
        while i*batch_size <= len(self.train_m):
            complete = []
            incomplete = []
            m = []
            delta = []
            last_values = []
            for j in range((i-1)*batch_size,i*batch_size):
                complete.append(self.train_complete[j])
                incomplete.append(self.train_incomplete[j])
                m.append(self.train_m[j])
                delta.append(self.train_delta[j])
                last_values.append(self.train_last[j])
            yield incomplete, complete, m, delta, last_values
            i+=1

    def next_val(self, batch_size):
        i = 1
        while i*batch_size <= len(self.val_m):
            complete = []
            incomplete = []
            m = []
            delta = []
            last_values = []
            for j in range((i-1)*batch_size,i*batch_size):
                complete.append(self.val_complete[j])
                incomplete.append(self.val_incomplete[j])
                m.append(self.val_m[j])
                delta.append(self.val_delta[j])
                last_values.append(self.val_last[j])
            yield incomplete, complete, m, delta, last_values
            i+=1


    def next_test(self, batch_size):
        i = 1
        while i*batch_size <= len(self.test_m):
            complete = []
            incomplete = []
            m = []
            delta = []
            last_values = []
            for j in range((i-1)*batch_size,i*batch_size):
                complete.append(self.test_complete[j])
                incomplete.append(self.test_incomplete[j])
                m.append(self.test_m[j])
                delta.append(self.test_delta[j])
                last_values.append(self.test_last[j])
            yield incomplete, complete, m, delta, last_values
            i+=1

if __name__ == '__main__':
    dt = Read()
    #print(dt.data[0])
    #print(dt.data[23])
    #print(len(dt.data_set))
    #print(dt.data_set[0])
    #print(sample_M(3,3,3,0.7))
    #print(dt.complete_data[0][0])
    #print(dt.incomplete_data[0][0])
    #print(dt.m[0][0])
    #print(dt.delta[0][0])
    #print(dt.lastvalues[0][0])
    count = 0
    for incomplete, complete, m, delta, last_values in dt.next_test(16):
        count += 1
        ic = incomplete[0]
        co = complete[0]
        mm = m[0]
        for i in range(len(ic)):
            for j in range(len(ic[i])):
                if ic[i][j] != co[i][j] *mm[i][j]:
                    print("false")
    print(count)
