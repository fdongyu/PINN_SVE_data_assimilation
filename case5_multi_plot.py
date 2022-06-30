#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 15:15:34 2021

@author: feng779
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from datetime import datetime
import time
from pyDOE import lhs
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
#print(tf.__version__)
import pickle
import os
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from SVE_module_dynamic_h import SVE as SVE_stardard
from SVE_module_dynamic_h_mff_ts import SVE as SVE_mff

import pdb


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

def add_noise(insignal):
    """
    https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
    """

    target_snr_db = 500
    # Calculate signal power and convert to dB 
    sig_avg = np.mean(insignal)
    sig_avg_db = 10 * np.log10(sig_avg)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg), len(insignal))
    # Noise up the original signal
    sig_noise = insignal + noise
    return sig_noise

if __name__ == "__main__": 
    
    """
    step one: read model output
    """
    hdf_filename = 'HEC-RAS/case5/MixedFlow.p02.hdf'
    hf = h5py.File(hdf_filename,'r') 
    
    attrs = hf['Geometry']['Cross Sections']['Attributes'][:]
    staid = []
    eles = []
    reach_len = []
    for attr in attrs:
        staid.append(attr[2].decode('utf-8'))
        eles.append(attr[14])
        reach_len.append(attr[6])

    coor = np.cumsum(np.array(reach_len[:-1]))
    coor = [0] + coor.tolist()
    coor = coor[::-1]
    eles = np.array(eles)
    slope = np.gradient( eles, coor)
     
    water_surface = hf['Results']['Unsteady']['Output']["/Results/Unsteady/Output"]['Output Blocks']['Base Output']["Unsteady Time Series"]["Cross Sections"]['Water Surface'][:]
    velocity_total = hf['Results']['Unsteady']['Output']["/Results/Unsteady/Output"]['Output Blocks']['Base Output']["Unsteady Time Series"]["Cross Sections"]['Velocity Total'][:]
    Timestamp = hf['Results']['Unsteady']['Output']["/Results/Unsteady/Output"]['Output Blocks']['Base Output']["Unsteady Time Series"]['Time Date Stamp'][:]
    time_model = time_convert(Timestamp)
    
    water_depth = water_surface - eles[None,:]
    b = 10 # channel width
    
    warmup_step = 1
    time_model = time_model[warmup_step:]
    velocity_total = velocity_total[warmup_step:]
    water_depth = water_depth[warmup_step:]
    eles = eles - eles[-1]
    
    Nt = water_depth.shape[0]
    Nx = water_depth.shape[1]
    
    Nt_train = water_depth.shape[0]
    Nf_train = 70000  # This is not used, all collocation points are used
    layers = [2] + 4*[1*64] + [2]
    
    t = np.arange(Nt_train)[:,None]
    x = np.array(coor[::-1])[:,None]
    u_exact = velocity_total[:Nt_train,:]
    h_exact = water_depth[:Nt_train,:]
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = u_exact.flatten()[:,None]    
    h_star = h_exact.flatten()[:,None]  
    
    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    ## IC multi time steps
    tsteps = [0] + [Nt-1]

    for i, tstep in enumerate(tsteps):
        xx1_ = np.hstack((X[tstep:tstep+1,:].T, T[tstep:tstep+1,:].T)) 
        hh1_ = add_noise(h_exact[tstep:tstep+1,:].T)
        if i == 0:
            xx1 = xx1_
            hh1 = hh1_
        else:
            xx1 = np.vstack((xx1, xx1_))
            hh1 = np.vstack((hh1, hh1_))

    xx2 = np.hstack((X[:,0:1], T[:,0:1]))   ## upstrm BC
    uu2 = u_exact[:,0:1]
    hh2 = h_exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))   ## dnstrm BC
    uu3 = u_exact[:,-1:]
    hh3 = h_exact[:,-1:]
    
    X_h_IC = xx1
    h_IC = hh1
    X_u_BC = np.vstack([xx2, xx3])
    X_h_BC = np.vstack([xx2, xx3])
    u_BC = np.vstack([uu2, uu3])
    h_BC = np.vstack([hh2, hh3])

    useObs = True 
    ## obs velocity
    ind_obs_u = [118, 134]
    t_obs_u = np.array([])
    x_obs_u = np.array([])
    u_obs = np.array([])
    for iobs in ind_obs_u:
        t_obs_u = np.append( t_obs_u, t.flatten() )
        x_obs_u = np.append( x_obs_u, np.ones(Nt_train)*x[iobs] )
        u_obs = np.append( u_obs, add_noise(u_exact[:Nt_train, iobs]) )
    X_u_obs = np.vstack([x_obs_u, t_obs_u]).T
    u_obs = u_obs[:,None]

    ## obs water depth
    ind_obs_h = [118,134]
    t_obs_h = np.array([])
    x_obs_h = np.array([])
    h_obs = np.array([])
    for iobs in ind_obs_h:
        t_obs_h = np.append( t_obs_h, t.flatten() )
        x_obs_h = np.append( x_obs_h, np.ones(Nt_train)*x[iobs] )
        h_obs = np.append( h_obs, add_noise(h_exact[:Nt_train, iobs]) )
    X_h_obs = np.vstack([x_obs_h, t_obs_h]).T
    h_obs = h_obs[:,None]

    X_f_train = X_star
    slope = np.hstack([np.array(slope) for _ in range(Nt_train)])[:,None]

    exist_mode = 2
    saved_path = 'saved_model/case5/PINN_SVE.pickle'
    weight_path = 'saved_model/case5/weights.out'
    # Training
    model1 = SVE_stardard(X_h_IC,
                X_u_BC, X_h_BC,
                X_u_obs, X_h_obs,
                X_f_train,
                h_IC,
                u_BC, h_BC,
                u_obs,h_obs,
                layers,
                lb, ub, slope, b,
                X_star, u_star, h_star,
                ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path,
                useObs=useObs) 

    # Test data
    Nt_test = Nt_train
    N_test = Nt_test * Nx    ## Nt_test x Nx
    X_test = X_star[:N_test,:]
    x_test = X_test[:,0:1]
    t_test = X_test[:,1:2]
    u_test = u_star[:N_test,:]
    h_test = h_star[:N_test,:]
    
    # Prediction
    u_pred1, h_pred1 = model1.predict(x_test, t_test)
    error_h1 = np.linalg.norm(h_test-h_pred1,2)/np.linalg.norm(h_test,2)
    print('Error h: %e' % (error_h1))

    rmse_h1 = np.sqrt(((h_test - h_pred1) ** 2).mean())
    print('RMSE h: %.3f m' % rmse_h1 )

    u_pred1 = u_pred1.reshape([Nt_test, Nx])
    h_pred1 = h_pred1.reshape([Nt_test, Nx])
    u_test = u_test.reshape([Nt_test, Nx])
    h_test = h_test.reshape([Nt_test, Nx])

    ## linear interpolation here
    ind_obs_interp = [0]+ind_obs_h+[h_test.shape[-1]-1]
    x_obs_interp = x.flatten()[ind_obs_interp]
    # add noise
    h_test_noise = np.zeros_like(h_test)
    for i in range(h_test.shape[1]):
        if i in ind_obs_h:
            h_test_noise[:,i] = add_noise(h_test[:,i])
        else:
            h_test_noise[:,i] = h_test[:,i]

    h_interp = np.zeros_like(h_test)
    h_interp[:,0]  = h_test[:,0]
    h_interp[:,-1] = h_test[:,-1]
    for i in range(h_test.shape[0]):
        if i != 0 or i != h_test.shape[0]-1:
            f = interp1d(x_obs_interp, h_test_noise[i,ind_obs_interp])
            h_interp[i,:] = f(x.flatten())

    error_h_interp = np.linalg.norm(h_test-h_interp,2)/np.linalg.norm(h_test,2)
    print('Error h (interp): %e' % (error_h_interp))

    rmse_h_interp = np.sqrt(((h_test - h_interp) ** 2).mean())
    print('RMSE h (interp): %.3f m' % rmse_h_interp )

    
    hours = np.arange(len(time_model[:Nt_test]))
    x = x[::-1]  ## reverse the channel for plotting
    xx, tt = np.meshgrid(x.flatten(), hours)

    factor = 0.3048   # ft to m 
    xx *= factor
    x  *= factor
    u_test *= factor
    u_pred1 *= factor
    h_test *= factor
    h_pred1 *= factor
    h_interp *= factor
    eles *= factor

    plt.rcParams.update({'font.size': 16})
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True)
    axes = axes.ravel()

    levels = np.linspace(0, 5, 9)
    cs = axes[0].contourf(xx, tt, h_test[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[0].set_ylabel('Time (h)')
    axes[0].set_xlabel('Distance upstream (m)')
    axes[0].set_title('Reference')

    cs = axes[1].contourf(xx, tt, h_pred1[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[1].set_xlabel('Distance upstream (m)')
    axes[1].set_title('Prediction (PINN)')

    cs = axes[2].contourf(xx, tt, h_interp[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[2].set_xlabel('Distance upstream (m)')
    axes[2].set_title('Prediction (interpolation)')

    axes = axes.ravel()
    for i in range(len(axes)):
        axes[i].text(0.05, 0.9, '{}'.format(labels[i]), fontsize=16, transform=axes[i].transAxes)
    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(cs, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=14)
    cb.ax.yaxis.offsetText.set_fontsize(14)
    cb.set_label('Water depth (m)', fontsize=14)

    plt.tight_layout()
    plt.savefig('figures/case5/contour.png')
    plt.close()

    tlist = [20, 50, 100, 200]
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots( 2, 2, figsize=(15, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    for k in range(len(tlist)):
        axes[k].plot(x, h_test[tlist[k],:]+eles, 'ok', label='reference')
        axes[k].plot(x, h_pred1[tlist[k],:]+eles, '-b', linewidth=2, label='PINN')
        axes[k].plot(x, h_interp[tlist[k],:]+eles, '-g', linewidth=2, label='interpolation')
        axes[k].fill_between(x.flatten(), eles, color='0.7')
        axes[k].text(0.85, 0.85, 't={} h'.format(int(tlist[k])), fontsize=15, transform=axes[k].transAxes)
        if k in [0,2]:
            axes[k].set_ylabel('Water stage (m)')
        axes[k].set_ylim([0,5])
        axes[k].grid()
        if k in [2,3]:
            axes[k].set_xlabel('Distance upstream (m)')
        if k == 0:
            axes[k].legend(loc=2,prop={'size': 12.5})

    plt.tight_layout()
    plt.savefig('figures/case5/along_channel.png')
    plt.close()
