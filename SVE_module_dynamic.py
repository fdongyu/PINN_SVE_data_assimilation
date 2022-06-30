#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 15:15:34 2021

@author: feng779
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from datetime import datetime
import time
from pyDOE import lhs
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
print(tf.__version__)
import pickle
import os

import warnings
warnings.filterwarnings("ignore")

#tf.config.run_functions_eagerly(True)

import pdb

np.random.seed(1234)
tf.set_random_seed(1234)

class SVE:
    
    DTYPE=tf.float32
    # Initialize the class
    def __init__(self, X_IC,
                      X_BC,
                      X_obs,
                      X_f, 
                      h_IC,
                      h_BC,
                      h_obs, 
                      layers,
                      lb, ub, S, n, u,
                      X_star, h_star,
                      ExistModel=0, uhDir='', wDir='', useObs=True):

        # Count for callback function
        self.count=0

        self.lb = lb
        self.ub = ub
        self.S = S  ## channel slope
        self.n = tf.constant(n, dtype=self.DTYPE)  ## manning n
        self.u = tf.constant(u, dtype=self.DTYPE)  ## constant velocity
        self.useObs = useObs
        
        # test data
        self.X_star = X_star
        self.h_star = h_star

        # Adaptive re-weighting constant
        self.beta = 0.9
        self.adaptive_constant_bcs_val = np.array(1.0)
        self.adaptive_constant_ics_val = np.array(1.0)
        self.adaptive_constant_obs_val = np.array(1.0)
        
        self.x_IC = X_IC[:,0:1]
        self.t_IC = X_IC[:,1:2]

        self.x_BC = X_BC[:,0:1]
        self.t_BC = X_BC[:,1:2]

        self.x_obs = X_obs[:,0:1]
        self.t_obs = X_obs[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.h_IC = h_IC
        self.h_BC = h_BC
        self.h_obs = h_obs
        
        # layers
        self.layers = layers
        
        # initialize NN
        if ExistModel == 0 :
            self.weights, self.biases = self.initialize_NN(layers)
        else:
            print("Loading uh NN ...")
            self.weights, self.biases = \
                self.load_NN(uhDir, layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # placeholders for data on velocities (inside the domain)
        self.x_IC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_IC.shape[1]])
        self.t_IC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_IC.shape[1]])
        self.h_IC_tf = tf.placeholder(self.DTYPE, shape=[None, self.h_IC.shape[1]])

        self.x_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_BC.shape[1]])
        self.t_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_BC.shape[1]])
        self.h_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.h_BC.shape[1]])

        self.x_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_obs.shape[1]])
        self.t_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_obs.shape[1]])
        self.h_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.h_obs.shape[1]])
        
        
        self.x_f_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])  

        if ExistModel == 0:
            self.adaptive_constant_bcs_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_bcs_val.shape)
            self.adaptive_constant_ics_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_ics_val.shape)
            self.adaptive_constant_obs_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_obs_val.shape)
        elif ExistModel in [1,2]:
            print ('Loading adaptive weights ...')
            if self.useObs:
                self.adaptive_constant_bcs_tf, \
                self.adaptive_constant_ics_tf, \
                self.adaptive_constant_obs_tf = self.load_weight(wDir)
                print("constant_bcs_val: {:.3f}, constant_ics_val: {:.3f}, constant_obs_val: {:.3f}".format(self.adaptive_constant_bcs_tf, self.adaptive_constant_ics_tf, self.adaptive_constant_obs_tf))
            else:
                self.adaptive_constant_bcs_tf, \
                self.adaptive_constant_ics_tf = self.load_weight(wDir)
                print("constant_bcs_val: {:.3f}, constant_ics_val: {:.3f}".format(self.adaptive_constant_bcs_tf, self.adaptive_constant_ics_tf))
        
        # physics informed neural networks
        self.h_pred = self.net_h(self.x_f_tf, self.t_f_tf)
        self.h_IC_pred = self.net_h(self.x_IC_tf, self.t_IC_tf) 
        self.h_BC_pred = self.net_h(self.x_BC_tf, self.t_BC_tf)
        if self.useObs:
            self.h_obs_pred = self.net_h(self.x_obs_tf, self.t_obs_tf)
        self.eq1_pred, self.eq2_pred = self.net_f(self.x_f_tf, self.t_f_tf) 
                                    
        # loss
        self.loss_f_c  = tf.reduce_mean(tf.square(self.eq1_pred)) ## continuity
        self.loss_f_m  = tf.reduce_mean(tf.square(self.eq2_pred)) ## momentum
        self.loss_f    = self.loss_f_c + self.loss_f_m

        self.loss_BC = tf.reduce_mean(tf.square(self.h_BC_tf - self.h_BC_pred))
        self.loss_BC = self.adaptive_constant_bcs_tf * self.loss_BC

        self.loss_IC = tf.reduce_mean(tf.square(self.h_IC_tf - self.h_IC_pred))
        self.loss_IC = self.adaptive_constant_ics_tf * self.loss_IC        

        self.loss = self.loss_f + self.loss_BC + self.loss_IC

        if self.useObs:
            self.loss_obs = tf.reduce_mean(tf.square(self.h_obs_tf - self.h_obs_pred))
            self.loss_obs = self.adaptive_constant_obs_tf * self.loss_obs
            self.loss += self.loss_obs
        

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0e-10,
                                                                           'gtol' : 0.000001})

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-4
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       5000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, global_step=self.global_step)

        ## Loss logger
        self.loss_f_c_log = []
        self.loss_f_m_log = []
        self.loss_BC_log  = []
        self.loss_IC_log  = []
        self.loss_obs_log = []
        self.l2_error_log = []

        # Generate dicts for gradients storage
        self.dict_gradients_res_layers = self.generate_grad_dict(self.layers)
        self.dict_gradients_bcs_layers = self.generate_grad_dict(self.layers)
        self.dict_gradients_ics_layers = self.generate_grad_dict(self.layers)
        if self.useObs:
            self.dict_gradients_obs_layers = self.generate_grad_dict(self.layers)

        # Gradients Storage
        self.grad_res = []
        self.grad_bcs = []
        self.grad_ics = []
        self.grad_obs = []
        for i in range(len(self.layers)-1):
            self.grad_res.append(tf.gradients(self.loss_f, self.weights[i])[0])
            self.grad_bcs.append(tf.gradients(self.loss_BC, self.weights[i])[0])
            self.grad_ics.append(tf.gradients(self.loss_IC, self.weights[i])[0])
            if self.useObs:
                self.grad_obs.append(tf.gradients(self.loss_obs, self.weights[i])[0])

        self.adpative_constant_bcs_list = []
        self.adpative_constant_bcs_log = []
        self.adpative_constant_ics_list = []
        self.adpative_constant_ics_log = []
        self.adpative_constant_obs_list = []
        self.adpative_constant_obs_log = []
        
        for i in range(len(self.layers) - 1):
            self.adpative_constant_bcs_list.append(
                tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_bcs[i])))
            self.adpative_constant_ics_list.append(
                tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_ics[i])))
            if self.useObs:
                self.adpative_constant_obs_list.append(
                    tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_obs[i])))

        self.adaptive_constant_bcs = tf.reduce_max(tf.stack(self.adpative_constant_bcs_list))
        self.adaptive_constant_ics = tf.reduce_max(tf.stack(self.adpative_constant_ics_list))
        if self.useObs:
            self.adaptive_constant_obs = tf.reduce_max(tf.stack(self.adpative_constant_obs_list))

        init = tf.global_variables_initializer()
        
        self.sess.run(init)


    def generate_grad_dict(self, layers):
        num = len(layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict
    
    # Save gradients during training
    def save_gradients(self, tf_dict):
        num_layers = len(self.layers)
        for i in range(num_layers - 1):
            if self.useObs:
                grad_res_value, grad_bcs_value, grad_ics_value, grad_obs_value = \
                self.sess.run([self.grad_res[i], self.grad_bcs[i], self.grad_ics[i], self.grad_obs[i]], feed_dict=tf_dict)
            else:
                grad_res_value, grad_bcs_value, grad_ics_value =\
                self.sess.run([self.grad_res[i], self.grad_bcs[i], self.grad_ics[i]], feed_dict=tf_dict)

            # save gradients of loss_r and loss_u
            self.dict_gradients_res_layers['layer_' + str(i + 1)].append(grad_res_value.flatten())
            self.dict_gradients_bcs_layers['layer_' + str(i + 1)].append(grad_bcs_value.flatten())
            self.dict_gradients_ics_layers['layer_' + str(i + 1)].append(grad_ics_value.flatten())
            if self.useObs:
                self.dict_gradients_obs_layers['layer_' + str(i + 1)].append(grad_obs_value.flatten())
        return None
                               
        
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=self.DTYPE), dtype=self.DTYPE)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=self.DTYPE)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_h(self, x, t):
        X = 2.0*(tf.concat([x, t], 1) - self.lb)/(self.ub - self.lb) - 1.0
        h = self.neural_net(X, self.weights, self.biases)
        
        return h
         
    def net_f(self, x_f, t_f):
        X_f = 2.0*(tf.concat([x_f, t_f], 1) - self.lb)/(self.ub - self.lb) - 1.0

        h = self.neural_net(X_f, self.weights, self.biases)
        h_t = tf.gradients(h, t_f)[0]
        h_x = tf.gradients(h, x_f)[0]
                
        eq1 = self.fun_r_mass(h_t, h_x)
        eq2 = self.fun_r_momentum(h, h_x)
           
        return eq1, eq2
        
    def fun_r_mass(self, h_t, h_x):
        
        return h_t + self.u * h_x

    def fun_r_momentum(self, h, h_x):
        h = tf.clip_by_value(h, clip_value_min=1e-5, clip_value_max=50)

        return h_x + self.S - (self.n*self.n * self.u*self.u) / tf.pow( tf.pow(h, 4), 1/3.)

    def callback_obs(self, loss, loss_f_c, loss_f_m, loss_BC, loss_IC, loss_obs):
        self.count = self.count+1
        print('{} th iterations, Loss: {:.3e}, Loss_f_c: {:.3e}, Loss_f_m: {:.3e}'.format(self.count, loss, loss_f_c, loss_f_m))
        self.loss_f_c_log.append(loss_f_c)
        self.loss_f_m_log.append(loss_f_m) 
        self.loss_BC_log.append(loss_BC)
        self.loss_IC_log.append(loss_IC)
        self.loss_obs_log.append(loss_obs)

    def callback(self, loss, loss_f_c, loss_f_m, loss_BC, loss_IC):
        self.count = self.count+1
        print('{} th iterations, Loss: {:.3e}, Loss_f_c: {:.3e}, Loss_f_m: {:.3e}'.format(self.count, loss, loss_f_c, loss_f_m))
        self.loss_f_c_log.append(loss_f_c)
        self.loss_f_m_log.append(loss_f_m)
        self.loss_BC_log.append(loss_BC)
        self.loss_IC_log.append(loss_IC)

    def train(self, num_epochs):
        
        tf_dict = {self.x_IC_tf: self.x_IC, self.t_IC_tf: self.t_IC, self.h_IC_tf: self.h_IC, 
                   self.x_BC_tf: self.x_BC, self.t_BC_tf: self.t_BC, self.h_BC_tf: self.h_BC, 
                   self.x_obs_tf: self.x_obs, self.t_obs_tf: self.t_obs, self.h_obs_tf: self.h_obs,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f, 
                   self.adaptive_constant_bcs_tf: self.adaptive_constant_bcs_val,
                   self.adaptive_constant_ics_tf: self.adaptive_constant_ics_val,
                   self.adaptive_constant_obs_tf: self.adaptive_constant_obs_val,
                   }
        
        for it in range(num_epochs):
            
            start_time = time.time()
            self.sess.run(self.train_op_Adam, tf_dict)
                
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                learning_rate = self.sess.run(self.learning_rate)
                print('It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e'
                         %(it, loss_value, elapsed, learning_rate))

                h_pred = self.predict(self.X_star[:,0:1], self.X_star[:,1:2])
                
                error_h = np.linalg.norm(self.h_star - h_pred, 2) / np.linalg.norm(self.h_star, 2)

                if self.useObs: 
                    loss_BC, loss_IC, loss_obs, loss_f_c, loss_f_m = \
                        self.sess.run([self.loss_BC, self.loss_IC, self.loss_obs, self.loss_f_c, self.loss_f_m], tf_dict)
                    print ('Loss_BC: %.3e, Loss_IC: %.3e, Loss_obs: %.3e, Loss_f_c: %.3e, Loss_f_m: %.3e, Error h: %.3e'
                            %(loss_BC, loss_IC, loss_obs, loss_f_c, loss_f_m, error_h))
                else:
                    loss_BC, loss_IC, loss_f_c, loss_f_m = \
                        self.sess.run([self.loss_BC, self.loss_IC, self.loss_f_c, self.loss_f_m], tf_dict)
                    print ('Loss_BC: %.3e, Loss_IC: %.3e, Loss_f_c: %.3e, Loss_f_m: %.3e, Error h: %.3e'
                            %(loss_BC, loss_IC, loss_f_c, loss_f_m, error_h))
                
                self.loss_f_c_log.append(loss_f_c)
                self.loss_f_m_log.append(loss_f_m)
                self.loss_BC_log.append(loss_BC)
                self.loss_IC_log.append(loss_IC)
                if self.useObs:
                    self.loss_obs_log.append(loss_obs)
                self.l2_error_log.append(error_h)

                # Compute the adaptive constant
                adaptive_constant_bcs_val = self.sess.run(self.adaptive_constant_bcs, tf_dict)
                self.adaptive_constant_bcs_val = adaptive_constant_bcs_val * \
                                            (1.0 - self.beta) + self.beta * self.adaptive_constant_bcs_val
                self.adpative_constant_bcs_log.append(self.adaptive_constant_bcs_val) 

                adaptive_constant_ics_val = self.sess.run(self.adaptive_constant_ics, tf_dict)
                self.adaptive_constant_ics_val = adaptive_constant_ics_val * \
                                            (1.0 - self.beta) + self.beta * self.adaptive_constant_ics_val
                self.adpative_constant_ics_log.append(self.adaptive_constant_ics_val)

                if self.useObs:                  
                    adaptive_constant_obs_val = self.sess.run(self.adaptive_constant_obs, tf_dict)
                    self.adaptive_constant_obs_val = adaptive_constant_obs_val * \
                                                (1.0 - self.beta) + self.beta * self.adaptive_constant_obs_val
                    self.adpative_constant_obs_log.append(self.adaptive_constant_obs_val)

                if self.useObs:
                    print("constant_bcs_val: {:.3f}, constant_ics_val: {:.3f}, constant_obs_val: {:.3f}".format(self.adaptive_constant_bcs_val, self.adaptive_constant_ics_val, self.adaptive_constant_obs_val))
                else:
                    print("constant_bcs_val: {:.3f}, constant_ics_val: {:.3f}".format(self.adaptive_constant_bcs_val, self.adaptive_constant_ics_val))

                start_time = time.time()

            # Store gradients
            if it % 10000 == 0:
                self.save_gradients(tf_dict)
                print ("Gradients information stored ...")

    def train_bfgs(self):

        tf_dict = {self.x_IC_tf: self.x_IC, self.t_IC_tf: self.t_IC, self.h_IC_tf: self.h_IC,
                   self.x_BC_tf: self.x_BC, self.t_BC_tf: self.t_BC, self.h_BC_tf: self.h_BC,
                   self.x_obs_tf: self.x_obs, self.t_obs_tf: self.t_obs, self.h_obs_tf: self.h_obs,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        if self.useObs:
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.loss_f_c, self.loss_f_m, self.loss_BC, self.loss_IC, self.loss_obs],
                                    loss_callback=self.callback_obs)
        else:
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.loss_f_c, self.loss_f_m, self.loss_BC, self.loss_IC],
                                    loss_callback=self.callback)


    def predict(self, x_star, t_star):
        
        tf_dict = {self.x_f_tf: x_star, self.t_f_tf: t_star}

        h_star = self.sess.run(self.h_pred, tf_dict)
        
        return h_star
    
    def save_NN(self, filepath):

        weights = self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        
        with open(filepath, 'wb') as f:
            pickle.dump([weights, biases], f)
            print("Save u h NN parameters successfully...")

    def load_NN(self, filepath, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(filepath, 'rb') as f:
            uh_weights, uh_biases = pickle.load(f)

            # Stored model must has the same # of layers
            assert num_layers == (len(uh_weights)+1)

            for num in range(0, num_layers - 1):
                weights.append( tf.Variable(uh_weights[num]) )
                biases.append( tf.Variable(uh_biases[num]) )
            print(" - Load NN parameters successfully...")
        return weights, biases

    def save_weight(self, filepath):
        if self.useObs:
            weight_array = np.vstack([np.array(self.adpative_constant_bcs_log),
                                      np.array(self.adpative_constant_ics_log),
                                      np.array(self.adpative_constant_obs_log)])
        else:
            weight_array = np.vstack([np.array(self.adpative_constant_bcs_log),
                                      np.array(self.adpative_constant_ics_log)])
        np.savetxt(filepath, weight_array.T, fmt='%1.4e')
  
    def load_weight(self, filepath):
        weight_array = np.loadtxt(filepath)
        weight_array = weight_array.T

        if self.useObs:
            return weight_array[0,-1], weight_array[1,-1], weight_array[2,-1]
        else:
            return weight_array[0,-1], weight_array[1,-1]
        


def time_convert(intime):
    """
    function to convert the time from string to datetime
    """
    Nt = intime.shape[0]
    outtime = []
    for t in range(Nt):
        timestr = intime[t].decode('utf-8')
        outtime.append(datetime.strptime(timestr, '%d%b%Y %H:%M:%S'))
    return outtime
