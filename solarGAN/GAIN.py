# -*- coding: utf-8 -*-  

'''
AQ_GAIN.py
Written by Jinsung Yoon
Modified by Yonghong Luo
Generative Adversarial Imputation Networks (GAIN) Implementation on MNIST
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN.pdf
Appendix Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN_Supp.pdf
Contact: jsyoon0823@g.ucla.edu
该文件主要用于直接测量填充准确度，aq数据用来回归的在另外一个文件中
'''

#%% Packages
from __future__ import print_function
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
import argparse
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import os
#from tqdm import tqdm
from data import read

#%% System Parameters
# 1. Mini batch size
mb_size = 8
# 2. Missing rate
#p_miss = 0.5
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Imput Dim (Fixed)
Dim = 24*39
# 6. No
Train_No = 55000
Test_No = 10000

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--missing-rate', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--n-steps', type=int, default=24)
args = parser.parse_args()
p_miss = args.missing_rate 

#%% Data Input
# 用dataset.next_train_data来计算MSE
dataset = read.Read(missing_rate = args.missing_rate, gap = args.n_steps)

# X
#trainX, _ = mnist.train.next_batch(Train_No) 
#testX, _  = mnist.test.next_batch(Test_No) 

# Mask Vector and Hint Vector Generation
# M应该为我自己生成,不要随机生成
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A > p
    C = 1.*B
    return C
  
#trainM = sample_M(Train_No, Dim, p_miss)
#testM = sample_M(Test_No, Dim, p_miss)

def unzip(data):
    new_data=[]
    for oneclass in data:
        #oneclass:48*31
        newclass=[]
        for one in oneclass:
            newclass.extend(one)
        new_data.append(newclass)
    return new_data

#%% Necessary Functions
# 1. Xavier Initialization Definition
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)
    
# 2. Plot (4 x 4 subfigures)
def plot(samples):
    fig = plt.figure(figsize = (5,5))
    gs = gridspec.GridSpec(5,5)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28), cmap='Greys_r')
        
    return fig
   
'''
GAIN Consists of 3 Components
- Generator
- Discriminator
- Hint Mechanism
'''   
   
#%% GAIN Architecture   
   
#%% 1. Input Placeholders
# 1.1. Data Vector
X = tf.placeholder(tf.float32, shape = [None, Dim])
X_complete = tf.placeholder(tf.float32, shape = [None, Dim])
# 1.2. Mask Vector 
M = tf.placeholder(tf.float32, shape = [None, Dim])
# 1.3. Hint vector
H = tf.placeholder(tf.float32, shape = [None, Dim])
# 1.4. Random Noise Vector
Z = tf.placeholder(tf.float32, shape = [None, Dim])

#%% 2. Discriminator
D_W1 = tf.Variable(xavier_init([Dim*2, 256]))     # Data + Hint as inputs
D_b1 = tf.Variable(tf.zeros(shape = [256]))

D_W2 = tf.Variable(xavier_init([256, 128]))
D_b2 = tf.Variable(tf.zeros(shape = [128]))

D_W3 = tf.Variable(xavier_init([128, Dim]))
D_b3 = tf.Variable(tf.zeros(shape = [Dim]))       # Output is multi-variate

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

#%% 3. Generator
G_W1 = tf.Variable(xavier_init([Dim*2, 256]))     # Data + Mask as inputs (Random Noises are in Missing Components)
G_b1 = tf.Variable(tf.zeros(shape = [256]))

G_W2 = tf.Variable(xavier_init([256, 128]))
G_b2 = tf.Variable(tf.zeros(shape = [128]))

G_W3 = tf.Variable(xavier_init([128, Dim]))
G_b3 = tf.Variable(tf.zeros(shape = [Dim]))

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

#%% GAIN Function

#%% 1. Generator
def generator(x,z,m):
    inp = m * x + (1-m) * z  # Fill in random noise on the missing values
    inputs = tf.concat(axis = 1, values = [inp,m])  # Mask + Data Concatenate
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output
    
    return G_prob
    
#%% 2. Discriminator
def discriminator(x, m, g, h):
    inp = m * x + (1-m) * g  # Replace missing values to the imputed values
    inputs = tf.concat(axis = 1, values = [inp,h])  # Hint + Data Concatenate
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output
    
    return D_prob

#%% 3. Others
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 1., size = [m, n])        

def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

#%% Structure
G_sample = generator(X,Z,M)
D_prob = discriminator(X, M, G_sample, H)

#%% Loss
D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8)) * 2
G_loss1 = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8)) / tf.reduce_mean(1-M)
MSE_train_loss = tf.reduce_sum((M * X - M * G_sample)**2) / tf.reduce_sum(M)

D_loss = D_loss1
G_loss = G_loss1  + alpha * MSE_train_loss 

#%% MSE Performance metric 不能直接这么算！ 因为丢失的部分都是0.。。。。。。要用原始数据集来算
MSE_test_loss = tf.reduce_sum(((1-M) * X_complete - (1-M)*G_sample)**2)/ tf.reduce_sum(1-M) 

#%% Solver
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#%%
# Output Initialization
#if not os.path.exists('Multiple_Impute_out1/'):
#    os.makedirs('Multiple_Impute_out1/')
    
# Iteration Initialization
i = 1

test_losses = []
#%% Start Iterations
for epoch in range(args.epochs):
    
    train_losses = []
    for incomplete, complete, m, delta, last_values in dataset.next_train(mb_size):
    #%% Inputs 合并成一个，拉平时间
        X_mb = np.array(unzip(incomplete))
        X_mb_complete = np.array(unzip(complete))

        Z_mb = sample_Z(mb_size, Dim) 
        M_mb = np.array(unzip(m))
        H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
        H_mb = M_mb * H_mb1
        
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
        
        _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb, H: H_mb, X_complete:X_mb_complete})
        _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
                feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb, H: H_mb, X_complete:X_mb_complete})
    
        #%% Intermediate Losses
        #print('Train_loss: {:.4}'.format(MSE_train_loss_curr))
        #print('Test_loss: {:.4}'.format(MSE_test_loss_curr))
        #print()
        train_losses.append(MSE_train_loss_curr)
        #test_losses.append(MSE_test_loss_curr)

    print("mean train loss: %.4f" %(sum(train_losses)/len(train_losses)))

for incomplete, complete, m, delta, last_values in dataset.next_test(mb_size):
#%% Inputs 合并成一个，拉平时间
    X_mb = np.array(unzip(incomplete))
    X_mb_complete = np.array(unzip(complete))

    Z_mb = sample_Z(mb_size, Dim) 
    M_mb = np.array(unzip(m))
    H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
    H_mb = M_mb * H_mb1
    
    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
    
   # _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb, H: H_mb, X_complete:X_mb_complete})
    MSE_test_loss_curr = sess.run(MSE_test_loss,
            feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb, H: H_mb, X_complete:X_mb_complete})

    test_losses.append(MSE_test_loss_curr)
print("test loss: %.4f" %(sum(test_losses)/len(test_losses)))
#print(test_losses)
