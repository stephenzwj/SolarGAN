# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:54:53 2018

@author: yonghong, luo
"""
from __future__ import print_function
import sys
sys.path.append("..")
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import argparse
import numpy as np
from data import read
import os
import WGAN_GRUI 

"""main"""
def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--impute-iter', type=int, default=100)
    parser.add_argument('--pretrain-epoch', type=int, default=10)
    parser.add_argument('--g-loss-lambda',type=float,default=0.1)
    parser.add_argument('--beta1',type=float,default=0.9)
    parser.add_argument('--lr', type=float, default=0.0001)
    #when l=0.001, pretrain_loss decreases rapidly
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--n-inputs', type=int, default=39)
    parser.add_argument('--n-steps', type=int, default=12)
    parser.add_argument('--n-hidden-units', type=int, default=32)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--missing-rate', type=float, default=0.5)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory name to save training logs')
    #0 false 1 true
    parser.add_argument('--isBatch-normal',type=int,default=1)
    parser.add_argument('--disc-iters',type=int,default=1)
    args = parser.parse_args()
    
    if args.isBatch_normal==0:
            args.isBatch_normal=False
    if args.isBatch_normal==1:
            args.isBatch_normal=True
    #make the max step length of two datasett the same
    min_mse = 1.0
    min_para = ""
    ff = open("result" + str(args.missing_rate), "a")
    epochs=[20,22,25,27,30]
    pres = [12,15,17,20,22]
    g_loss_lambdas=[0.02,0.05,0.1,0.15,0.2,0.3]
    lrs = [0.007] #0.004,0.005,0.006,0.007已经跑完了
    for pre in pres:
        for e in epochs:
            for g_l in g_loss_lambdas:
                for lr in lrs:
                    args.epoch=e
                    args.pretrain_epoch = pre
                    args.g_loss_lambda=g_l
                    args.lr = lr
                    tf.reset_default_graph()
                    data_set = read.Read(missing_rate = args.missing_rate, gap = args.n_steps)
                    config = tf.ConfigProto() 
                    config.gpu_options.allow_growth = True 
                    with tf.Session(config=config) as sess:
                        gan = WGAN_GRUI.WGAN(sess,
                                args=args,
                                datasets=data_set,
                                )

                        # build graph
                        gan.build_model()
                
                        # show network architecture
                        #show_all_variables()
                
                        # launch the graph in a session
                        gan.train()
                        print(" [*] Training finished!")
                    
                        mse, paras = gan.imputation()
                        if mse < min_mse:
                            min_mse = mse 
                            min_para = paras
                        ff.write(str(min_mse) + "," + min_para + "\r\n")
                        ff.flush()
                        print(" [*] Train dataset Imputation finished!")
                        print("now min mse: " + str(min_mse))
                        print("now min para: " + min_para)
            
                    tf.reset_default_graph()
    print(min_mse)
    print(min_para)
    ff.close()
if __name__ == '__main__':
    main()
