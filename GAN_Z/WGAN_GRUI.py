#-*- coding: utf-8 -*-
from __future__ import division
import os
import math
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from ops import bn
#from utils import *
from GRUI import mygru_cell

"""
D输入标准化， 不要m 填充0
G输入去掉m,只有delta
g 没有每次累加z
"""

def xavier_init(sizes):
    in_dim = sizes[0]
    xavier_stddev = 1. / math.sqrt(in_dim / 2.)
    #np.random.normal(loc = 0.0, scale = 0.1, size = (self.batch_size, self.z_dim))
    return np.random.normal(size = sizes, loc = 0.0,  scale = xavier_stddev)

def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A >= p
    C = 1.*B
    return C


class WGAN(object):
    model_name = "WGAN_no_mask_"     # name for checkpoint

    def __init__(self, sess, args, datasets):
        self.sess = sess
        self.isbatch_normal=args.isBatch_normal
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.lr = args.lr                 
        self.epoch = args.epoch     
        self.batch_size = args.batch_size
        self.n_inputs = args.n_inputs                 # MNIST data input (img shape: 28*28)
        self.n_steps = args.n_steps                             
        self.n_hidden_units = args.n_hidden_units        # neurons in hidden layer
        self.pretrain_epoch=args.pretrain_epoch
        self.impute_iter=args.impute_iter
        self.g_loss_lambda=args.g_loss_lambda
        self.missing_rate = args.missing_rate
	self.model_name += str(self.missing_rate)       	
 
        self.datasets=datasets
        self.z_dim = args.z_dim         # dimension of noise-vector
        
        # WGAN_GP parameter
        self.disc_iters = args.disc_iters     # The number of critic iterations for one-step of generator

        # train
        self.learning_rate = args.lr
        self.beta1 = args.beta1
        if "1.5" in tf.__version__ or "1.7" in tf.__version__ :
            self.grud_cell_d = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_g = mygru_cell.MyGRUCell15(self.n_hidden_units)
        elif "1.4" in tf.__version__:
            self.grud_cell_d = mygru_cell.MyGRUCell4(self.n_hidden_units)
            self.grud_cell_g = mygru_cell.MyGRUCell4(self.n_hidden_units)
        elif "1.2" in tf.__version__:
            self.grud_cell_d = mygru_cell.MyGRUCell2(self.n_hidden_units)
            self.grud_cell_g = mygru_cell.MyGRUCell2(self.n_hidden_units)
        # test
        self.sample_num = 64  # number of generated images to be saved

        self.num_batches = len(datasets.m) // self.batch_size

      
    def pretrainG(self, z, X, M, Delta, Lastvalues, X_lengths, Keep_prob, reuse=False):
        
        with tf.variable_scope("g_enerator", reuse=reuse):
            
            """
            the rnn cell's variable scope is defined by tensorflow,
            if we want to update rnn cell's weights, the variable scope must contains 'g_' or 'd_'
            
            """
            
            wr_h=tf.get_variable("g_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.contrib.layers.xavier_initializer())
            w_out= tf.get_variable("g_w_out",shape=[self.n_hidden_units, self.n_inputs],initializer=tf.contrib.layers.xavier_initializer())
            
            br_h= tf.get_variable("g_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.0))
            b_out= tf.get_variable("g_b_out",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.0))
            w_z=tf.get_variable("g_w_z",shape=[self.z_dim,self.n_inputs],initializer=tf.contrib.layers.xavier_initializer())
            b_z=tf.get_variable("g_b_z",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.0))
            
            z = tf.reshape(z,[-1,self.z_dim])
            zz = tf.matmul(z,w_z)+b_z
            X = tf.reshape(X, [-1, self.n_inputs]) 
            X = X + zz 
            Delta=tf.reshape(Delta,[-1,self.n_inputs])
            
            rth= tf.matmul(Delta, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            
            X=tf.concat([X,rth],1)
            
            X_in = tf.reshape(X, [-1, self.n_steps, self.n_inputs+self.n_hidden_units])
         
            init_state = self.grud_cell_g.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            outputs, final_state = tf.nn.dynamic_rnn(self.grud_cell_g, X_in, \
                                initial_state=init_state,\
                                sequence_length=X_lengths,
                                time_major=False)
            #outputs: batch_size*n_steps*n_hiddensize
            outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
            out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
            out_predict=tf.reshape(out_predict,[-1,self.n_steps,self.n_inputs])
            return out_predict


    def discriminator(self, X, M, DeltaPre, X_lengths,Keep_prob, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("d_iscriminator", reuse=reuse):
            
            wr_h=tf.get_variable("d_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.contrib.layers.xavier_initializer())
            w_out= tf.get_variable("d_w_out",shape=[self.n_hidden_units, 1],initializer=tf.contrib.layers.xavier_initializer())
            br_h= tf.get_variable("d_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.0))
            b_out= tf.get_variable("d_b_out",shape=[1, ],initializer=tf.constant_initializer(0.0))
          
           
            M=tf.reshape(M,[-1,self.n_inputs])
            X = tf.reshape(X, [-1, self.n_inputs])
            DeltaPre=tf.reshape(DeltaPre,[-1,self.n_inputs])
           
            
            rth= tf.matmul(DeltaPre, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            # add noise
            #X=X+np.random.standard_normal(size=(self.batch_size*self.n_steps, self.n_inputs))/100 
            X=tf.concat([X,rth],1)
              
            X_in = tf.reshape(X, [self.batch_size, self.n_steps , self.n_inputs+self.n_hidden_units])
            
            init_state = self.grud_cell_d.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            outputs, final_state = tf.nn.dynamic_rnn(self.grud_cell_d, X_in, \
                                initial_state=init_state,\
                                sequence_length=X_lengths,
                                time_major=False)
         
            # final_state:batch_size*n_hiddensize
            # 不能用最后一个，应该用第length个  之前用了最后一个，所以输出无论如何都是b_out
            out_logit=tf.matmul(tf.nn.dropout(final_state,Keep_prob), w_out) + b_out
            out =tf.nn.sigmoid(out_logit)    #选取最后一个 output
            return out,out_logit

    def generator(self, z, Keep_prob, is_training=True, reuse=False):
        # x,delta,n_steps
        # z :[self.batch_size, self.z_dim]
        # first feed noize in rnn, then feed the previous output into next input
        # or we can feed noize and previous output into next input in future version
        with tf.variable_scope("g_enerator", reuse=reuse):
            #gennerate 
            
            wr_h=tf.get_variable("g_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.contrib.layers.xavier_initializer())
            w_out= tf.get_variable("g_w_out",shape=[self.n_hidden_units, self.n_inputs],initializer=tf.contrib.layers.xavier_initializer())
            br_h= tf.get_variable("g_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.0))
            b_out= tf.get_variable("g_b_out",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.0))
            w_z=tf.get_variable("g_w_z",shape=[self.z_dim,self.n_inputs],initializer=tf.contrib.layers.xavier_initializer())
            b_z=tf.get_variable("g_b_z",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.0))
            
            #self.times=tf.reshape(self.times,[self.batch_size,self.n_steps,self.n_inputs])
            #change z's dimension
            # batch_size*z_dim-->batch_size*n_inputs
            x=tf.matmul(z,w_z)+b_z
            x=tf.reshape(x,[-1,self.n_inputs])
            delta_zero=tf.constant(0.0,shape=[self.batch_size,self.n_inputs])
            #delta_normal=tf.constant(48.0*60.0/self.gen_length,shape=[self.batch_size,self.n_inputs])
            #delta:[batch_size,1,n_inputs]
            

            # combine X_in
            rth= tf.matmul(delta_zero, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            x=tf.concat([x,rth],1)
            
            X_in = tf.reshape(x, [-1, 1, self.n_inputs+self.n_hidden_units])
            
            init_state = self.grud_cell_g.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            #z=tf.reshape(z,[self.batch_size,1,self.z_dim])
            seq_len=tf.constant(1,shape=[self.batch_size])
            
            outputs, final_state = tf.nn.dynamic_rnn(self.grud_cell_g, X_in, \
                                initial_state=init_state,\
                                sequence_length=seq_len,
                                time_major=False)
            init_state=final_state
            #outputs: batch_size*1*n_hidden
            outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
            # full connect
            out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
            randomms = sample_M(self.batch_size, self.n_inputs, 1)
            #1 代表randoms全是0
            out_predict = self.get_time(self.lastvalues, 0) * randomms + (1 - randomms) * out_predict
            out_predict=tf.reshape(out_predict,[-1,1,self.n_inputs])
            
            total_result=tf.multiply(out_predict,1.0)
            
            for i in range(1,self.n_steps):
                out_predict=tf.reshape(out_predict,[self.batch_size,self.n_inputs])
                #输出加上noise z
                #out_predict=out_predict+tf.matmul(z,w_z)+b_z
                #
                delta_normal=tf.reshape(self.imputed_deltaPre[:,i:(i+1),:],[self.batch_size,self.n_inputs])
                rth= tf.matmul(delta_normal, wr_h)+br_h
                rth=math_ops.exp(-tf.maximum(0.0,rth))
                x=tf.concat([out_predict,rth],1)
                X_in = tf.reshape(x, [-1, 1, self.n_inputs+self.n_hidden_units])
                
                outputs, final_state = tf.nn.dynamic_rnn(self.grud_cell_g, X_in, \
                            initial_state=init_state,\
                            sequence_length=seq_len,
                            time_major=False)
                init_state=final_state
                outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
                out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
                randomms = sample_M(self.batch_size, self.n_inputs, 1)
                out_predict = self.get_time(self.lastvalues, i) * randomms + (1 - randomms) * out_predict
                out_predict=tf.reshape(out_predict,[-1,1,self.n_inputs])
                total_result=tf.concat([total_result,out_predict],1)
            
            #delta:[batch_size,,n_inputs]
        
            if self.isbatch_normal:
                with tf.variable_scope("g_bn", reuse=tf.AUTO_REUSE):
                    total_result=bn(total_result,is_training=is_training, scope="g_bn_imple")
            
            

            return total_result
    
    def get_time(self, x, i):
        return tf.reshape(x[:,i:(i+1),:],[self.batch_size,self.n_inputs])



    def impute(self):
        with tf.variable_scope("impute", reuse=tf.AUTO_REUSE):
            #z_need_tune=tf.get_variable("z_needtune",shape=[self.batch_size, self.n_steps, self.z_dim],initializer=tf.contrib.layers.xavier_initializer() )
            z_need_tune=tf.get_variable("z_needtune",shape=[self.batch_size, self.n_steps, self.z_dim],initializer=tf.random_normal_initializer()  )
            return z_need_tune
            
    def build_model(self):
        
        self.keep_prob = tf.placeholder(tf.float32) 
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.complete = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.deltaPre = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.lastvalues = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.n_steps, self.z_dim], name='z')

        self.x_lengths=tf.constant(self.n_steps,shape=[self.batch_size,])
        
        self.imputed_deltaPre=tf.constant(1.0,  shape=[self.batch_size,self.n_steps,self.n_inputs])
        self.imputed_m = tf.constant(1.0,  shape=[self.batch_size,self.n_steps,self.n_inputs])
        
        

        """ Loss Function """

        Pre_out=self.pretrainG(self.z, self.lastvalues, self.m, self.deltaPre,self.lastvalues, self.x_lengths,self.keep_prob, reuse=False)
        
        self.pretrain_loss=tf.reduce_sum(tf.square(tf.multiply(Pre_out,self.m)-self.x)) / tf.reduce_sum(self.m)

        self.pre_acc = tf.reduce_sum(((1-self.m) * self.complete  - (1-self.m)*Pre_out)**2)/ tf.reduce_sum(1-self.m)
        
        D_real, D_real_logits = self.discriminator(self.lastvalues, self.m, self.deltaPre, self.x_lengths,self.keep_prob, reuse=False)

        #g_x = self.generator(self.z,self.keep_prob, is_training=True, reuse=True)
        # 
        #D_fake, D_fake_logits = self.discriminator(g_x, self.imputed_m, self.imputed_deltaPre, self.x_lengths, self.keep_prob,reuse=True)
        
        D_fake, D_fake_logits = self.discriminator(Pre_out, self.imputed_m, self.imputed_deltaPre, self.x_lengths, self.keep_prob,reuse=True)
        """
        impute loss
        """
        self.z_need_tune=self.impute()
        
        #impute_out = self.generator(self.z_need_tune,self.keep_prob, is_training=False, reuse=True)
        impute_out = self.pretrainG(self.z_need_tune, self.lastvalues, self.m, self.deltaPre,self.lastvalues, self.x_lengths,self.keep_prob, reuse=True)
        
        
        impute_fake, impute_fake_logits = self.discriminator(impute_out,self.imputed_m, self.imputed_deltaPre, self.x_lengths, self.keep_prob, reuse=True)
        
        # loss for imputation
        
        self.impute_loss=tf.reduce_sum((impute_out*self.m-self.m*self.x)**2)/tf.reduce_sum(self.m) - self.g_loss_lambda*tf.reduce_mean(impute_fake_logits)
        #self.impute_loss=tf.reduce_mean(tf.square(tf.multiply(impute_out,self.m)-self.x))
        
        self.impute_out=impute_out
        
        # get loss for discriminator
        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = self.g_loss_lambda*tf.reduce_mean(D_fake_logits) - tf.reduce_sum(tf.square(tf.multiply(Pre_out, self.m)-self.x))/tf.reduce_sum(self.m)


        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = - d_loss_fake 
        #- tf.reduce_sum(tf.square(tf.multiply(Pre_out, self.m)-self.x))/tf.reduce_sum(self.m)

        self.acc = tf.reduce_sum(((1-self.m) * self.complete  - (1-self.m)*self.impute_out)**2)/ tf.reduce_sum(1-self.m)
        

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        z_vars = [self.z_need_tune]
        
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # this code have used batch normalization, so the upside line should be executed
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                        .minimize(self.d_loss, var_list=d_vars)
            #self.d_optim=self.optim(self.learning_rate, self.beta1,self.d_loss,d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*self.disc_iters, beta1=self.beta1) \
                        .minimize(self.g_loss, var_list=g_vars)
            #self.g_optim=self.optim(self.learning_rate, self.beta1,self.g_loss,g_vars)
            self.g_pre_optim=tf.train.AdamOptimizer(self.learning_rate*4,beta1=self.beta1) \
                        .minimize(self.pretrain_loss,var_list=g_vars)
        self.impute_optim=tf.train.AdamOptimizer(self.learning_rate*1,beta1=self.beta1) \
                    .minimize(self.impute_loss,var_list=z_vars)
    
        
        

        #clip weight
        self.clip_all_vals = [p.assign(tf.clip_by_value(p, -0.6, 0.6)) for p in t_vars]
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.6, 0.6)) for p in d_vars]
        self.clip_G = [p.assign(tf.clip_by_value(p, -0.6, 0.6)) for p in g_vars]
        
        
        """" Testing """

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        g_pretrain_loss_sum=tf.summary.scalar("g_pretrain_loss", self.pretrain_loss)
        # final summary operations
        self.impute_sum=tf.summary.scalar("impute_loss", self.impute_loss)
        self.g_sum = g_loss_sum
        self.g_pretrain_sum=tf.summary.merge([g_pretrain_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum,d_loss_fake_sum, d_loss_sum])
        
    def optim(self,learning_rate,beta,loss,var):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta)
        grads = optimizer.compute_gradients(loss,var_list=var)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
        train_op = optimizer.apply_gradients(grads)
        return train_op
    def pretrain(self, start_epoch,counter,start_time):
        
        if start_epoch < self.pretrain_epoch:
            #todo
            for epoch in range(start_epoch, self.pretrain_epoch):
            # get batch data
                idx=0
                    
                for incomplete, complete, m, delta, last_values in self.datasets.next(self.batch_size):
                    # pretrain
                    batch_z = np.random.normal(loc = 0.0, scale = 1, size = (self.batch_size,self.n_steps, self.z_dim))
                    #batch_z = xavier_init(sizes = [self.batch_size, self.n_steps, self.z_dim])
                    _ = self.sess.run(self.clip_all_vals)
                    acc, _, summary_str, p_loss = self.sess.run([self.pre_acc, self.g_pre_optim, self.g_pretrain_sum, self.pretrain_loss],
                                                   feed_dict={
                                                              self.z: batch_z,
                                                              self.x: incomplete,
                                                              self.m: m,
                                                              self.deltaPre: delta,
                                                              self.lastvalues: last_values,
                                                              self.complete: complete,
                                                              self.keep_prob: 0.7})
                    self.writer.add_summary(summary_str, counter)
    
    
                    counter += 1
    
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, pretrain_loss: %.8f, acc : %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, p_loss, acc))
                    idx+=1
                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0

                # save model
                #调好之后再保存
                #if epoch%10==0:
                #    self.save(self.checkpoint_dir, counter)


    def train(self):

        # graph inputs for visualize training results
        self.sample_z = np.random.standard_normal(size=(self.batch_size , self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name+'/'+self.model_dir)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            #start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            start_batch_id=0
            #counter = checkpoint_counter
            counter=start_epoch*self.num_batches
            print(" [*] Load SUCCESS")
            return 
        else:
            # initialize all variables
            tf.global_variables_initializer().run()
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        
        self.pretrain(start_epoch,counter,start_time)
        if start_epoch < self.pretrain_epoch:
            start_epoch=self.pretrain_epoch
        
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            idx=0
            for incomplete, complete, m, delta, last_values in self.datasets.next(self.batch_size):    
                batch_z = np.random.normal(loc = 0.0, scale = 1, size = (self.batch_size,self.n_steps, self.z_dim))
                #batch_z = xavier_init(sizes = [self.batch_size, self.n_steps, self.z_dim])
                #_ = self.sess.run(self.clip_D)
                _ = self.sess.run(self.clip_all_vals)
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], 
                                           feed_dict={
                                               self.z: batch_z,
                                               self.x: incomplete,
                                               self.m: m,
                                               self.deltaPre: delta,
                                               self.lastvalues: last_values,
                                               self.complete: complete,
                                               self.keep_prob: 0.7})
                self.writer.add_summary(summary_str, counter)
                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f, counter:%4d" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, g_loss, counter))

                # update D network
                if counter%self.disc_iters==0:
                    _ = self.sess.run(self.clip_all_vals)
                    acc, _, summary_str, d_loss = self.sess.run([self.pre_acc, self.d_optim, self.d_sum, self.d_loss],
                                                   feed_dict={
                                                       self.z: batch_z,
                                                       self.x: incomplete,
                                                       self.m: m,
                                                       self.deltaPre: delta,
                                                       self.lastvalues: last_values,
                                                       self.complete: complete,
                                                       self.keep_prob: 0.7})
                    self.writer.add_summary(summary_str, counter)
                    #batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f,counter:%4d, acc: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, counter, acc))
                    #debug 

                counter += 1


                # save training results for every 300 steps
                if np.mod(counter, 300) == 0 :
                    fake_x, = self.sess.run([self.impute_out],
                                               feed_dict={
                                                   self.z: batch_z,
                                                   self.x: incomplete,
                                                   self.m: m,
                                                   self.deltaPre: delta,
                                                   self.lastvalues: last_values,
                                                   self.complete: complete,
                                                   self.keep_prob: 0.5})
                    self.writeG_Samples("G_sample_x",counter,fake_x)
                    
                idx+=1
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

        
        self.save(self.checkpoint_dir, counter)

    def imputation(self):
        accs = []
        tf.variables_initializer([self.z_need_tune]).run()
        #是否shuffle无所谓，填充之后存起来，测试的时候用填充之后的数据再shuffle即可
        #训练数据集不能被batch_size整除剩下的部分，扔掉
        start_time = time.time()
        batchid=1
        impute_tune_time=1
        counter=1
        for incomplete, complete, m, delta, last_values in self.datasets.next(self.batch_size):    
            #self.z_need_tune=tf.assign(self.z_need_tune,tf.random_normal([self.batch_size,self.z_dim]))
            tf.variables_initializer([self.z_need_tune]).run()
            for i in range(0,self.impute_iter):
                _ = self.sess.run(self.clip_all_vals)
                _, impute_out, summary_str, impute_loss, acc = self.sess.run([self.impute_optim, self.impute_out, self.impute_sum, self.impute_loss, self.acc ], \
                                               feed_dict={
                                                   self.x: incomplete,
                                                   self.m: m,
                                                   self.deltaPre: delta,
                                                   self.lastvalues: last_values,
                                                   self.complete: complete,
                                                   self.keep_prob: 0.7})
                impute_tune_time+=1
                counter+=1
                if i == self.impute_iter - 1:
                    accs.append(acc)
                if counter%10==0:
                    print("Batchid: [%2d] [%4d/%4d] time: %4.4f, impute_loss: %.8f, acc: %.8f" \
                          % (batchid, impute_tune_time, self.impute_iter, time.time() - start_time, impute_loss, acc))
                    self.writer.add_summary(summary_str, counter/10)
            #imputed=tf.multiply((1-self.m),impute_out)+data_x
            accs.append(acc)
            batchid+=1
            impute_tune_time=1
        print(str(sum(accs)/len(accs)))
	return sum(accs)/len(accs),self.model_dir
    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.epoch,self.pretrain_epoch,
            self.disc_iters,
            self.batch_size, self.z_dim,
            self.lr,self.impute_iter,
            self.g_loss_lambda,
            self.beta1,self.missing_rate
            )


    def save_imputation(self,impute_out,batchid,data_x_lengths,data_times,data_y,isTrain):
        #填充后的数据全是n_steps长度！，但只有data_x_lengths才是可用的
        if isTrain:
            imputation_dir="imputation_train_results"
        else:
            imputation_dir="imputation_test_results"
        
        if not os.path.exists(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir)):
            os.makedirs(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir))
            
        #write imputed data
        resultFile=open(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir,\
                                     "batch"+str(batchid)+"x"),'w')
        for length in data_x_lengths:
            resultFile.writelines(str(length)+",")
        resultFile.writelines("\r\n")
        # impute_out:ndarray
        for oneSeries in impute_out:
            resultFile.writelines("begin\r\n")
            for oneClass in oneSeries:
                for i in oneClass.flat:
                    resultFile.writelines(str(i)+",")
                resultFile.writelines("\r\n")
            resultFile.writelines("end\r\n")
        resultFile.close()
        
        #write data_times data_times:list
        resultFile=open(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir,\
                                     "batch"+str(batchid)+"delta"),'w')
        for oneSeries in data_times:
            resultFile.writelines("begin\r\n")
            for oneClass in oneSeries:
                for i in oneClass:
                    resultFile.writelines(str(i)+",")
                resultFile.writelines("\r\n")
            resultFile.writelines("end\r\n")
        resultFile.close()
        
        #write y
        resultFile=open(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir,\
                                     "batch"+str(batchid)+"y"),'w')
        for oneSeries in data_y:
            #resultFile.writelines("begin\r\n")
            for oneClass in oneSeries:
                resultFile.writelines(str(oneClass)+",")
            resultFile.writelines("\r\n")
            #resultFile.writelines("end\r\n")
        resultFile.close()
        
    def writeG_Samples(self,filename,step,o):
        if not os.path.exists(os.path.join("G_results",\
                                     self.model_name,\
                                     self.model_dir)):
            os.makedirs(os.path.join("G_results",\
                                     self.model_name,\
                                     self.model_dir))
        resultFile=open(os.path.join("G_results",\
                                     self.model_name,\
                                     self.model_dir,\
                                     filename+str(step)),'w')
        for oneSeries in o:
            resultFile.writelines("begin\r\n")
            for oneClass in oneSeries:
                for i in oneClass.flat:
                    resultFile.writelines(str(i)+",")
                resultFile.writelines("\r\n")
            resultFile.writelines("end\r\n")
        resultFile.close()
    
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name, self.model_dir )

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
