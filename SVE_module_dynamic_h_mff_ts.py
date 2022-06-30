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
    def __init__(self, X_h_IC, 
                      X_u_BC, X_h_BC,
                      X_u_obs, X_h_obs,
                      X_f, 
                      h_IC,
                      u_BC, h_BC,
                      u_obs,h_obs, 
                      layers,
                      lb, ub, S, b, 
                      X_star, u_star, h_star, 
                      ExistModel=0, uhDir='', wDir='', wmffDir='', useObs=True):

        ## first layer is map to fourier space
        layers = layers[1:]

        # Count for callback function
        self.count=0

        self.lb = lb
        self.ub = ub
        self.S = S  ## channel slope
        self.b = tf.constant(b, dtype=self.DTYPE)  ## channel width
        self.useObs = useObs

        # test data
        self.X_star = X_star
        self.u_star = u_star
        self.h_star = h_star

        # Adaptive re-weighting constant
        self.beta = 0.9
        self.adaptive_constant_bcs_u_val = np.array(1.0)
        self.adaptive_constant_bcs_h_val = np.array(1.0)
        self.adaptive_constant_ics_u_val = np.array(1.0)
        self.adaptive_constant_ics_h_val = np.array(1.0)
        self.adaptive_constant_obs_u_val = np.array(1.0)
        self.adaptive_constant_obs_h_val = np.array(1.0)
        
        self.x_h_IC = X_h_IC[:,0:1]
        self.t_h_IC = X_h_IC[:,1:2]

        self.x_u_BC = X_u_BC[:,0:1]
        self.t_u_BC = X_u_BC[:,1:2]
        self.x_h_BC = X_h_BC[:,0:1]
        self.t_h_BC = X_h_BC[:,1:2]

        self.x_u_obs = X_u_obs[:,0:1]
        self.t_u_obs = X_u_obs[:,1:2]
        self.x_h_obs = X_h_obs[:,0:1]
        self.t_h_obs = X_h_obs[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.h_IC = h_IC
        self.u_BC = u_BC
        self.h_BC = h_BC
        self.u_obs = u_obs
        self.h_obs = h_obs
        
        # Initialize multi-scale Fourier features
        self.W1_t = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=self.DTYPE) * 0.25,
                               dtype=self.DTYPE, trainable=False)
        self.W2_t = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=self.DTYPE) * 10,
                               dtype=self.DTYPE, trainable=False)
        #self.W1_x = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=self.DTYPE) * 1.0,
        #                       dtype=self.DTYPE, trainable=False)
        self.W1_x = tf.Variable(tf.ones([1, layers[0] // 2], dtype=self.DTYPE) * 1.0,
                               dtype=self.DTYPE, trainable=False)
        


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
        self.x_u_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_u_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])
        self.x_h_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_h_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])

        self.x_h_IC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_h_IC.shape[1]])
        self.t_h_IC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_h_IC.shape[1]])
        self.h_IC_tf = tf.placeholder(self.DTYPE, shape=[None, self.h_IC.shape[1]])

        self.x_u_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_u_BC.shape[1]])
        self.t_u_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_u_BC.shape[1]])
        self.u_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.u_BC.shape[1]])
        self.x_h_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_h_BC.shape[1]])
        self.t_h_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_h_BC.shape[1]])
        self.h_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.h_BC.shape[1]])

        self.x_u_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_u_obs.shape[1]])
        self.t_u_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_u_obs.shape[1]])
        self.u_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.u_obs.shape[1]])
        self.x_h_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_h_obs.shape[1]])
        self.t_h_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_h_obs.shape[1]])
        self.h_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.h_obs.shape[1]])
        
        
        self.x_f_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])  


        if ExistModel == 0:
            self.adaptive_constant_bcs_u_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_bcs_u_val.shape)
            self.adaptive_constant_bcs_h_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_bcs_h_val.shape)
            self.adaptive_constant_ics_u_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_ics_u_val.shape)
            self.adaptive_constant_ics_h_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_ics_h_val.shape)
            self.adaptive_constant_obs_u_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_obs_u_val.shape)
            self.adaptive_constant_obs_h_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_obs_h_val.shape)
        elif ExistModel in [1,2]:
            self.adaptive_constant_bcs_u_tf, \
            self.adaptive_constant_bcs_h_tf, \
            self.adaptive_constant_ics_h_tf, \
            self.adaptive_constant_obs_u_tf, \
            self.adaptive_constant_obs_h_tf = self.load_weight(wDir)
            print ('Loading adaptive weights ...')
            print("constant_bcs_u_val: {:.3f}, constant_bcs_h_val: {:.3f}, constant_ics_h_val: {:.3f}, constant_obs_u_val: {:.3f}, constant_obs_h_val: {:.3f}".format(self.adaptive_constant_bcs_u_tf, self.adaptive_constant_bcs_h_tf, self.adaptive_constant_ics_h_tf, self.adaptive_constant_obs_u_tf, self.adaptive_constant_obs_h_tf))
            
            self.W1_t, self.W2_t, self.W1_x = self.load_wmff(wmffDir)
        
        # physics informed neural networks
        self.u_pred, self.h_pred = self.net_uh(self.x_u_tf, self.t_u_tf)
        self.u_IC_pred, self.h_IC_pred = self.net_uh(self.x_h_IC_tf, self.t_h_IC_tf) 
        self.u_BC_pred, self.h_BC_pred = self.net_uh(self.x_u_BC_tf, self.t_u_BC_tf)
        if self.useObs:
            self.u_obs_pred, self.h_obs_pred = self.net_uh(self.x_h_obs_tf, self.t_h_obs_tf)
        self.eq1_pred, self.eq2_pred = self.net_f(self.x_f_tf, self.t_f_tf) 
                                    
        # loss
        self.loss_f_c  = tf.reduce_mean(tf.square(self.eq1_pred)) ## continuity
        self.loss_f_m  = tf.reduce_mean(tf.square(self.eq2_pred)) ## momentum
        self.loss_f    = self.loss_f_c + self.loss_f_m

        self.loss_BC_u = tf.reduce_mean(tf.square(self.u_BC_tf - self.u_BC_pred))
        self.loss_BC_h = tf.reduce_mean(tf.square(self.h_BC_tf - self.h_BC_pred))
        self.loss_BCs = self.adaptive_constant_bcs_u_tf * self.loss_BC_u \
                        + self.adaptive_constant_bcs_h_tf * self.loss_BC_h

        self.loss_IC_h = tf.reduce_mean(tf.square(self.h_IC_tf - self.h_IC_pred))
        self.loss_ICs = self.adaptive_constant_ics_h_tf * self.loss_IC_h        

        self.loss = self.loss_f + self.loss_BCs + self.loss_ICs

        if self.useObs:
            self.loss_obs_u = tf.reduce_mean(tf.square(self.u_obs_tf - self.u_obs_pred))
            self.loss_obs_h = tf.reduce_mean(tf.square(self.h_obs_tf - self.h_obs_pred))
            self.loss_obs = self.adaptive_constant_obs_u_tf * self.loss_obs_u \
                        + self.adaptive_constant_obs_h_tf * self.loss_obs_h
            self.loss += self.loss_obs
        

        # optimizers
        #self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
        #                                                        method = 'L-BFGS-B', 
        #                                                        options = {'maxiter': 50000,
        #                                                                   'maxfun': 50000,
        #                                                                   'maxcor': 50,
        #                                                                   'maxls': 50,
        #                                                                   'ftol' : 1.0 * np.finfo(float).eps})   
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0e-10,
                                                                           'gtol' : 0.000001})

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 5e-4
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       5000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, global_step=self.global_step)

        ## Loss logger
        self.loss_f_c_log = []
        self.loss_f_m_log = []
        self.loss_BC_u_log  = []
        self.loss_BC_h_log  = []
        self.loss_IC_h_log  = []
        self.loss_obs_u_log = []
        self.loss_obs_h_log = []
        self.l2_u_error_log = []
        self.l2_h_error_log = []

        # Generate dicts for gradients storage
        self.dict_gradients_res_layers = self.generate_grad_dict(layers)
        self.dict_gradients_bcs_u_layers = self.generate_grad_dict(layers)
        self.dict_gradients_bcs_h_layers = self.generate_grad_dict(layers)
        self.dict_gradients_ics_h_layers = self.generate_grad_dict(layers)
        if self.useObs:
            self.dict_gradients_obs_u_layers = self.generate_grad_dict(layers)
            self.dict_gradients_obs_h_layers = self.generate_grad_dict(layers)

        # Gradients Storage
        self.grad_res = []
        self.grad_bcs_u = []
        self.grad_bcs_h = []
        self.grad_ics_h = []
        self.grad_obs_u = []
        self.grad_obs_h = []
        for i in range(len(layers)-1):
            self.grad_res.append(tf.gradients(self.loss_f, self.weights[i])[0])
            self.grad_bcs_u.append(tf.gradients(self.loss_BC_u, self.weights[i])[0])
            self.grad_bcs_h.append(tf.gradients(self.loss_BC_h, self.weights[i])[0])
            self.grad_ics_h.append(tf.gradients(self.loss_IC_h, self.weights[i])[0])
            if self.useObs:
                self.grad_obs_u.append(tf.gradients(self.loss_obs_u, self.weights[i])[0])
                self.grad_obs_h.append(tf.gradients(self.loss_obs_h, self.weights[i])[0])

        self.adpative_constant_bcs_u_list = []
        self.adpative_constant_bcs_u_log = []
        self.adpative_constant_bcs_h_list = []
        self.adpative_constant_bcs_h_log = []
        self.adpative_constant_ics_h_list = []
        self.adpative_constant_ics_h_log = []
        self.adpative_constant_obs_u_list = []
        self.adpative_constant_obs_u_log = []
        self.adpative_constant_obs_h_list = []
        self.adpative_constant_obs_h_log = []

        for i in range(len(layers)-1):
            self.adpative_constant_bcs_u_list.append(
                tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_bcs_u[i])))
            self.adpative_constant_bcs_h_list.append(
                tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_bcs_h[i])))
            self.adpative_constant_ics_h_list.append(
                tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_ics_h[i])))
            if self.useObs:
                self.adpative_constant_obs_u_list.append(
                    tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_obs_u[i])))
                self.adpative_constant_obs_h_list.append(
                    tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_obs_h[i])))

        self.adaptive_constant_bcs_u = tf.reduce_max(tf.stack(self.adpative_constant_bcs_u_list))
        self.adaptive_constant_bcs_h = tf.reduce_max(tf.stack(self.adpative_constant_bcs_h_list))
        self.adaptive_constant_ics_h = tf.reduce_max(tf.stack(self.adpative_constant_ics_h_list))
        if self.useObs:
            self.adaptive_constant_obs_u = tf.reduce_max(tf.stack(self.adpative_constant_obs_u_list))
            self.adaptive_constant_obs_h = tf.reduce_max(tf.stack(self.adpative_constant_obs_h_list))

        init = tf.global_variables_initializer()
        
        self.sess.run(init)


    def generate_grad_dict(self, layers):
        num = len(layers)  
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict
    
    # Save gradients during training
    def save_gradients(self, tf_dict):
        num_layers = len(self.layers) -1
        for i in range(num_layers):
            if self.useObs:
                grad_res_value, grad_bcs_u_value, grad_bcs_h_value, grad_ics_h_value, grad_obs_u_value, grad_obs_h_value = \
                self.sess.run([self.grad_res[i], self.grad_bcs_u[i], self.grad_bcs_h[i], self.grad_ics_h[i], self.grad_obs_u[i], self.grad_obs_h[i]], feed_dict=tf_dict)
            else:
                grad_res_value, grad_bcs_u_value, grad_bcs_h_value, grad_ics_h_value =\
                self.sess.run([self.grad_res[i], self.grad_bcs_u[i], self.grad_bcs_h[i], self.grad_ics_h[i]], feed_dict=tf_dict)

            # save gradients of loss_r and loss_u
            self.dict_gradients_res_layers['layer_' + str(i + 1)].append(grad_res_value.flatten())
            self.dict_gradients_bcs_u_layers['layer_' + str(i + 1)].append(grad_bcs_u_value.flatten())
            self.dict_gradients_bcs_h_layers['layer_' + str(i + 1)].append(grad_bcs_h_value.flatten())
            self.dict_gradients_ics_h_layers['layer_' + str(i + 1)].append(grad_ics_h_value.flatten())
            if self.useObs:
                self.dict_gradients_obs_u_layers['layer_' + str(i + 1)].append(grad_obs_u_value.flatten())
                self.dict_gradients_obs_h_layers['layer_' + str(i + 1)].append(grad_obs_h_value.flatten())
        return None
                               
        
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-2):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=self.DTYPE), dtype=self.DTYPE)
            weights.append(W)
            biases.append(b)        

        W = self.xavier_init(size=[2 * layers[-2], layers[-1]])
        b = tf.Variable(tf.zeros([1, layers[-1]], dtype=self.DTYPE), dtype=self.DTYPE)
        weights.append(W)
        biases.append(b)

        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=self.DTYPE)
    
    def neural_net(self, X, weights, biases):
        #num_layers = len(weights) 
        num_layers = len(self.layers)
        
        H = X
        x = H[:,0:1]
        t = H[:,1:2]

        # Fourier feature encodings
        H1_t = tf.concat([tf.sin(tf.matmul(t, self.W1_t)),
                        tf.cos(tf.matmul(t, self.W1_t))], 1)

        H2_t = tf.concat([tf.sin(tf.matmul(t, self.W2_t)),
                        tf.cos(tf.matmul(t, self.W2_t))], 1)
        
        #H1_x = tf.concat([tf.sin(tf.matmul(x, self.W1_x)),
        #                tf.cos(tf.matmul(x, self.W1_x))], 1)
        H1_x = tf.concat([tf.matmul(x, self.W1_x),
                        tf.matmul(x, self.W1_x)], 1)

        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H1_t = tf.tanh(tf.add(tf.matmul(H1_t, W), b))
            H2_t = tf.tanh(tf.add(tf.matmul(H2_t, W), b))
            H1_x = tf.tanh(tf.add(tf.matmul(H1_x, W), b))

        # Merge the outputs by concatenation
        H1 = tf.multiply(H1_t, H1_x)
        H2 = tf.multiply(H2_t, H1_x)
        H = tf.concat([H1, H2], 1)

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_uh(self, x, t):
        X = 2.0*(tf.concat([x, t], 1) - self.lb)/(self.ub - self.lb) - 1.0
        #x_ = 2.0*(x - self.lb[0])/(self.ub[0]-self.lb[0]) - 1.0
        #t_ = (t - self.lb[1])/(self.ub[1]-self.lb[1])
        #X = tf.concat([x_, t_], 1)
        uh = self.neural_net(X, self.weights, self.biases)
        
        u = uh[:,0:1]
        h = uh[:,1:2]
        
        return u, h
         
    def net_f(self, x_f, t_f):
        X_f = 2.0*(tf.concat([x_f, t_f], 1) - self.lb)/(self.ub - self.lb) - 1.0
        #x_f_ = 2.0*(x_f - self.lb[0])/(self.ub[0]-self.lb[0]) - 1.0
        #t_f_ = (t_f - self.lb[1])/(self.ub[1]-self.lb[1])
        #X_f = tf.concat([x_f_, t_f_], 1)
        uh = self.neural_net(X_f, self.weights, self.biases)
        u = uh[:,0:1]
        h = uh[:,1:2]

        u_t = tf.gradients(u, t_f)[0]
        u_x = tf.gradients(u, x_f)[0]
        
        h_t = tf.gradients(h, t_f)[0]
        h_x = tf.gradients(h, x_f)[0]
        
        eq1 = self.fun_r_mass(u, h_t, h_x)
        eq2 = self.fun_r_momentum(u, h, u_t, u_x, h_x)
           
        return eq1, eq2
        
    def fun_r_mass(self, u, h_t, h_x):
        
        return h_t + u * h_x

    def fun_r_momentum(self, u, h, u_t, u_x, h_x):
        n = 0.015

        h = tf.clip_by_value(h, clip_value_min=1e-4, clip_value_max=50)
        R = self.b*h/(2*h+self.b)
        
        #return u_t + u*u_x + 9.81*h_x + 9.81* ( tf.math.divide_no_nan( n*n*tf.abs(u)*u , tf.pow( tf.square(R), 2./3) )  - self.S)
        return u_t + u*u_x + 9.81*h_x + 9.81* ( n*n*tf.abs(u)*u / tf.pow( tf.square(R), 2./3)   - self.S)

    def callback(self, loss, loss_f_c, loss_f_m, loss_BC_u, loss_BC_h, loss_IC_h, loss_obs_u, loss_obs_h):
        self.count = self.count+1
        print('{} th iterations, Loss: {:.3e}, Loss_f_c: {:.3e}, Loss_f_m: {:.3e}'.format(self.count, loss, loss_f_c, loss_f_m))
        self.loss_f_c_log.append(loss_f_c)
        self.loss_f_m_log.append(loss_f_m) 
        self.loss_BC_u_log.append(loss_BC_u)
        self.loss_BC_h_log.append(loss_BC_h)
        self.loss_IC_h_log.append(loss_IC_h)
        if self.useObs:
            self.loss_obs_u_log.append(loss_obs_u)
            self.loss_obs_h_log.append(loss_obs_h)

    def train(self, num_epochs):
        
        tf_dict = {self.x_h_IC_tf: self.x_h_IC, self.t_h_IC_tf: self.t_h_IC, self.h_IC_tf: self.h_IC, 
                   self.x_u_BC_tf: self.x_u_BC, self.t_u_BC_tf: self.t_u_BC, self.u_BC_tf: self.u_BC, 
                   self.x_h_BC_tf: self.x_h_BC, self.t_h_BC_tf: self.t_h_BC, self.h_BC_tf: self.h_BC,
                   self.x_u_obs_tf: self.x_u_obs, self.t_u_obs_tf: self.t_u_obs, self.u_obs_tf: self.u_obs,
                   self.x_h_obs_tf: self.x_h_obs, self.t_h_obs_tf: self.t_h_obs, self.h_obs_tf: self.h_obs,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f, 
                   self.adaptive_constant_bcs_u_tf: self.adaptive_constant_bcs_u_val,
                   self.adaptive_constant_bcs_h_tf: self.adaptive_constant_bcs_h_val,
                   self.adaptive_constant_ics_h_tf: self.adaptive_constant_ics_h_val,
                   self.adaptive_constant_obs_u_tf: self.adaptive_constant_obs_u_val,
                   self.adaptive_constant_obs_h_tf: self.adaptive_constant_obs_h_val,
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

                u_pred, h_pred = self.predict(self.X_star[:,0:1], self.X_star[:,1:2])
                error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                error_h = np.linalg.norm(self.h_star - h_pred, 2) / np.linalg.norm(self.h_star, 2)

                if self.useObs: 
                    loss_BC_u, loss_BC_h, loss_IC_h, loss_obs_u, loss_obs_h, loss_f_c, loss_f_m = \
                        self.sess.run([self.loss_BC_u, self.loss_BC_h, self.loss_IC_h, self.loss_obs_u, self.loss_obs_h, self.loss_f_c, self.loss_f_m], tf_dict)
                    print ('Loss_BC_u: %.3e, Loss_BC_h: %.3e, Loss_IC_h: %.3e, Loss_obs_u: %.3e, Loss_obs_h: %.3e, Loss_f_c: %.3e, Loss_f_m: %.3e, Error u: %.3e, Error h: %.3e'
                            %(loss_BC_u, loss_BC_h, loss_IC_h, loss_obs_u, loss_obs_h, loss_f_c, loss_f_m, error_u, error_h))
                else:
                    loss_BC_u, loss_BC_h, loss_IC_h, loss_f_c, loss_f_m = \
                        self.sess.run([self.loss_BC_u, self.loss_BC_h, self.loss_IC_h, self.loss_f_c, self.loss_f_m], tf_dict)
                    print ('Loss_BC_u: %.3e, Loss_BC_h: %.3e, Loss_IC_h: %.3e, Loss_f_c: %.3e, Loss_f_m: %.3e, Error u: %.3e, Error h: %.3e'
                            %(loss_BC_u, loss_BC_h, loss_IC_h, loss_f_c, loss_f_m, error_u, error_h))
                
                self.loss_f_c_log.append(loss_f_c)
                self.loss_f_m_log.append(loss_f_m)
                self.loss_BC_u_log.append(loss_BC_u)
                self.loss_BC_h_log.append(loss_BC_h)
                self.loss_IC_h_log.append(loss_IC_h)
                if self.useObs:
                    self.loss_obs_u_log.append(loss_obs_u)
                    self.loss_obs_h_log.append(loss_obs_h)
                self.l2_u_error_log.append(error_u)
                self.l2_h_error_log.append(error_h)

                # Compute the adaptive constant
                adaptive_constant_bcs_u_val = self.sess.run(self.adaptive_constant_bcs_u, tf_dict)
                self.adaptive_constant_bcs_u_val = adaptive_constant_bcs_u_val * \
                                            (1.0 - self.beta) + self.beta * self.adaptive_constant_bcs_u_val
                self.adpative_constant_bcs_u_log.append(self.adaptive_constant_bcs_u_val) 

                adaptive_constant_bcs_h_val = self.sess.run(self.adaptive_constant_bcs_h, tf_dict)
                self.adaptive_constant_bcs_h_val = adaptive_constant_bcs_h_val * \
                                            (1.0 - self.beta) + self.beta * self.adaptive_constant_bcs_h_val
                self.adpative_constant_bcs_h_log.append(self.adaptive_constant_bcs_h_val)

                adaptive_constant_ics_h_val = self.sess.run(self.adaptive_constant_ics_h, tf_dict)
                self.adaptive_constant_ics_h_val = adaptive_constant_ics_h_val * \
                                            (1.0 - self.beta) + self.beta * self.adaptive_constant_ics_h_val
                self.adpative_constant_ics_h_log.append(self.adaptive_constant_ics_h_val)

                if self.useObs:                  
                    adaptive_constant_obs_u_val = self.sess.run(self.adaptive_constant_obs_u, tf_dict)
                    self.adaptive_constant_obs_u_val = adaptive_constant_obs_u_val * \
                                                (1.0 - self.beta) + self.beta * self.adaptive_constant_obs_u_val
                    self.adpative_constant_obs_u_log.append(self.adaptive_constant_obs_u_val)

                    adaptive_constant_obs_h_val = self.sess.run(self.adaptive_constant_obs_h, tf_dict)
                    self.adaptive_constant_obs_h_val = adaptive_constant_obs_h_val * \
                                                (1.0 - self.beta) + self.beta * self.adaptive_constant_obs_h_val 
                    self.adpative_constant_obs_h_log.append(self.adaptive_constant_obs_h_val)

                if self.useObs:
                    print("constant_bcs_u_val: {:.3f}, constant_bcs_h_val: {:.3f}, constant_ics_h_val: {:.3f}, constant_obs_u_val: {:.3f}, constant_obs_h_val: {:.3f}".format(self.adaptive_constant_bcs_u_val, self.adaptive_constant_bcs_h_val, self.adaptive_constant_ics_h_val, self.adaptive_constant_obs_u_val, self.adaptive_constant_obs_h_val))
                else:
                    print("constant_bcs_u_val: {:.3f}, constant_bcs_h_val: {:.3f}, constant_ics_h_val: {:.3f}".format(self.adaptive_constant_bcs_u_val, self.adaptive_constant_bcs_h_val, self.adaptive_constant_ics_h_val))

                start_time = time.time()

            # Store gradients
            if it % 10000 == 0:
                self.save_gradients(tf_dict)
                print ("Gradients information stored ...")

    def train_bfgs(self):

        tf_dict = {self.x_h_IC_tf: self.x_h_IC, self.t_h_IC_tf: self.t_h_IC, self.h_IC_tf: self.h_IC,
                   self.x_u_BC_tf: self.x_u_BC, self.t_u_BC_tf: self.t_u_BC, self.u_BC_tf: self.u_BC,
                   self.x_h_BC_tf: self.x_h_BC, self.t_h_BC_tf: self.t_h_BC, self.h_BC_tf: self.h_BC,
                   self.x_u_obs_tf: self.x_u_obs, self.t_u_obs_tf: self.t_u_obs, self.u_obs_tf: self.u_obs,
                   self.x_h_obs_tf: self.x_h_obs, self.t_h_obs_tf: self.t_h_obs, self.h_obs_tf: self.h_obs,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        if self.useObs:
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.loss_f_c, self.loss_f_m, self.loss_BC_u, self.loss_BC_h, self.loss_IC_h, self.loss_obs_u, self.loss_obs_h],
                                    loss_callback=self.callback)
        else:
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.loss_f_c, self.loss_f_m, self.loss_BC_u, self.loss_BC_h, self.loss_IC_h],
                                    loss_callback=self.callback)


    def predict(self, x_star, t_star):
        
        tf_dict = {self.x_u_tf: x_star, self.t_u_tf: t_star,
                   self.x_h_tf: x_star, self.t_h_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        h_star = self.sess.run(self.h_pred, tf_dict)
        
        return u_star, h_star
    
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
            weight_array = np.vstack([np.array(self.adpative_constant_bcs_u_log),
                                      np.array(self.adpative_constant_bcs_h_log),
                                      np.array(self.adpative_constant_ics_h_log),
                                      np.array(self.adpative_constant_obs_u_log),
                                      np.array(self.adpative_constant_obs_h_log)])
        else:
            weight_array = np.vstack([np.array(self.adpative_constant_bcs_u_log),
                                      np.array(self.adpative_constant_bcs_h_log),
                                      np.array(self.adpative_constant_ics_h_log)])
        np.savetxt(filepath, weight_array.T, fmt='%1.4e')
  
    def load_weight(self, filepath):
        weight_array = np.loadtxt(filepath)
        weight_array = weight_array.T

        if self.useObs:
            return weight_array[0,-1], weight_array[1,-1], weight_array[2,-1], weight_array[3,-1], weight_array[4,-1]
        else:
            return weight_array[0,-1], weight_array[1,-1], weight_array[2,-1]

    def save_wmff(self, filepath):
        W1_t, W2_t, W1_x = self.sess.run([self.W1_t, self.W2_t, self.W1_x])
        wmff_array = np.vstack([W1_t, W2_t, W1_x])
        wmff_array = wmff_array.T
        np.savetxt(filepath, wmff_array, fmt='%1.6e')

    def load_wmff(self, filepath):
        wmff_array = np.loadtxt(filepath)
        wmff_array = wmff_array.T
        W1_t, W2_t, W1_x = [tf.convert_to_tensor(arg[:,None].T, dtype=self.DTYPE) for arg in wmff_array]
        return W1_t, W2_t, W1_x


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
