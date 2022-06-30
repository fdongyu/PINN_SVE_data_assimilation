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
    hdf_filename = 'HEC-RAS/case6/MixedFlow.p02.hdf'
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
    layers = [2] + 5*[1*64] + [2]
    
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

    X_f_train = X_star
    slope = np.hstack([np.array(slope) for _ in range(Nt_train)])[:,None]

    useObs = True
    exist_mode = 2

    #### 2 observations
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

    saved_path = 'saved_model/case6_mff_2obs/PINN_SVE.pickle'
    weight_path = 'saved_model/case6_mff_2obs/weights.out'
    wmff_path = 'saved_model/case6_mff_2obs/w_mff.out'
    # Training
    model_2obs = SVE_mff(X_h_IC,
                X_u_BC, X_h_BC,
                X_u_obs, X_h_obs,
                X_f_train,
                h_IC,
                u_BC, h_BC,
                u_obs,h_obs,
                layers,
                lb, ub, slope, b,
                X_star, u_star, h_star,
                ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path, wmffDir=wmff_path,
                useObs=useObs) 


    #### 3obs
    saved_path = 'saved_model/case6_mff_3obs/PINN_SVE.pickle'
    weight_path = 'saved_model/case6_mff_3obs/weights.out'
    wmff_path = 'saved_model/case6_mff_3obs/w_mff.out'
    # Training
    model_3obs = SVE_mff(X_h_IC,
                X_u_BC, X_h_BC,
                X_u_obs, X_h_obs,
                X_f_train,
                h_IC,
                u_BC, h_BC,
                u_obs,h_obs,
                layers,
                lb, ub, slope, b,
                X_star, u_star, h_star,
                ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path, wmffDir=wmff_path,
                useObs=useObs) 

    saved_path = 'saved_model/case6_mff_4obs/PINN_SVE.pickle'
    weight_path = 'saved_model/case6_mff_4obs/weights.out'
    wmff_path = 'saved_model/case6_mff_4obs/w_mff.out'
    # Training
    model_4obs = SVE_mff(X_h_IC,
                X_u_BC, X_h_BC,
                X_u_obs, X_h_obs,
                X_f_train,
                h_IC,
                u_BC, h_BC,
                u_obs,h_obs,
                layers,
                lb, ub, slope, b,
                X_star, u_star, h_star,
                ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path, wmffDir=wmff_path,
                useObs=useObs)

    saved_path = 'saved_model/case6_mff_5obs/PINN_SVE.pickle'
    weight_path = 'saved_model/case6_mff_5obs/weights.out'
    wmff_path = 'saved_model/case6_mff_5obs/w_mff.out'
    # Training
    model_5obs = SVE_mff(X_h_IC,
                X_u_BC, X_h_BC,
                X_u_obs, X_h_obs,
                X_f_train,
                h_IC,
                u_BC, h_BC,
                u_obs,h_obs,
                layers,
                lb, ub, slope, b,
                X_star, u_star, h_star,
                ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path, wmffDir=wmff_path,
                useObs=useObs)

    
    # Test data
    Nt_test = Nt_train
    N_test = Nt_test * Nx    ## Nt_test x Nx
    X_test = X_star[:N_test,:]
    x_test = X_test[:,0:1]
    t_test = X_test[:,1:2]
    h_test = h_star[:N_test,:]
    
    # Prediction
    u_pred_2obs, h_pred_2obs = model_2obs.predict(x_test, t_test)
    error_h_2obs = np.linalg.norm(h_test-h_pred_2obs,2)/np.linalg.norm(h_test,2)
    rmse_h_2obs = np.sqrt(((h_test - h_pred_2obs) ** 2).mean())

    u_pred_3obs, h_pred_3obs = model_3obs.predict(x_test, t_test)
    error_h_3obs = np.linalg.norm(h_test-h_pred_3obs,2)/np.linalg.norm(h_test,2)
    rmse_h_3obs = np.sqrt(((h_test - h_pred_3obs) ** 2).mean())

    u_pred_4obs, h_pred_4obs = model_4obs.predict(x_test, t_test)
    error_h_4obs = np.linalg.norm(h_test-h_pred_4obs,2)/np.linalg.norm(h_test,2)
    rmse_h_4obs = np.sqrt(((h_test - h_pred_4obs) ** 2).mean())

    u_pred_5obs, h_pred_5obs = model_5obs.predict(x_test, t_test)
    error_h_5obs = np.linalg.norm(h_test-h_pred_5obs,2)/np.linalg.norm(h_test,2)
    rmse_h_5obs = np.sqrt(((h_test - h_pred_5obs) ** 2).mean())
    
    h_pred_2obs = h_pred_2obs.reshape([Nt_test, Nx])
    h_pred_3obs = h_pred_3obs.reshape([Nt_test, Nx])
    h_pred_4obs = h_pred_4obs.reshape([Nt_test, Nx])
    h_pred_5obs = h_pred_5obs.reshape([Nt_test, Nx]) 
    h_test = h_test.reshape([Nt_test, Nx])

    def linear_downscale(ind_obs_in):
        # linear interpolation here 
        ind_obs_interp = [0]+ind_obs_in+[h_test.shape[-1]-1]
        x_obs_interp = x.flatten()[ind_obs_interp]
        # add noise
        h_test_noise = np.zeros_like(h_test)
        for i in range(h_test.shape[1]):
            if i in ind_obs_in:
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
        return h_interp

    ## linear interpolation here
    ind_obs_h_2obs = [118, 134]
    ind_obs_h_3obs = [25, 118, 134]
    ind_obs_h_4obs = [25, 50, 118, 134]
    ind_obs_h_5obs = [25, 50, 100, 118, 134]
    h_interp_2obs = linear_downscale(ind_obs_h_2obs) 
    h_interp_3obs = linear_downscale(ind_obs_h_3obs)
    h_interp_4obs = linear_downscale(ind_obs_h_4obs)
    h_interp_5obs = linear_downscale(ind_obs_h_5obs)

    error_h_interp_2obs = np.linalg.norm(h_test-h_interp_2obs,2)/np.linalg.norm(h_test,2)
    error_h_interp_3obs = np.linalg.norm(h_test-h_interp_3obs,2)/np.linalg.norm(h_test,2)
    error_h_interp_4obs = np.linalg.norm(h_test-h_interp_4obs,2)/np.linalg.norm(h_test,2)
    error_h_interp_5obs = np.linalg.norm(h_test-h_interp_5obs,2)/np.linalg.norm(h_test,2)

    rmse_h_interp_2obs = np.sqrt(((h_test - h_interp_2obs) ** 2).mean())
    rmse_h_interp_3obs = np.sqrt(((h_test - h_interp_3obs) ** 2).mean())
    rmse_h_interp_4obs = np.sqrt(((h_test - h_interp_4obs) ** 2).mean())
    rmse_h_interp_5obs = np.sqrt(((h_test - h_interp_5obs) ** 2).mean())

    print('2obs Error h: %e' % (error_h_2obs))
    print('2obs Error h (interp): %e' % (error_h_interp_2obs))
    print('2obs RMSE h: %.3f m' % (rmse_h_2obs))
    print('2obs RMSE h (interp): %.3f m' % (rmse_h_interp_2obs))

    print('3obs Error h: %e' % (error_h_3obs))    
    print('3obs Error h (interp): %e' % (error_h_interp_3obs))
    print('3obs RMSE h: %.3f m' % (rmse_h_3obs))
    print('3obs RMSE h (interp): %.3f m' % (rmse_h_interp_3obs))

    print('4obs Error h: %e' % (error_h_4obs))
    print('4obs Error h (interp): %e' % (error_h_interp_4obs))
    print('4obs RMSE h: %.3f m' % (rmse_h_4obs))
    print('4obs RMSE h (interp): %.3f m' % (rmse_h_interp_4obs))

    print('5obs Error h: %e' % (error_h_5obs))
    print('5obs Error h (interp): %e' % (error_h_interp_5obs))
    print('5obs RMSE h: %.3f m' % (rmse_h_5obs))
    print('5obs RMSE h (interp): %.3f m' % (rmse_h_interp_5obs))
    
    hours = np.arange(len(time_model[:Nt_test]))
    x = x[::-1]  ## reverse the channel for plotting
    xx, tt = np.meshgrid(x.flatten(), hours)

    factor = 0.3048   # ft to m 
    xx *= factor
    x  *= factor
    h_test *= factor
    h_pred_2obs *= factor
    h_pred_3obs *= factor
    h_pred_4obs *= factor	
    h_pred_5obs *= factor
    h_interp_2obs *= factor
    h_interp_3obs *= factor
    h_interp_4obs *= factor
    h_interp_5obs *= factor
    eles *= factor

    plt.rcParams.update({'font.size': 18})
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    fig = plt.figure(figsize=(10.5, 16))
    gs = gridspec.GridSpec(5, 4, hspace=0.08, wspace=0.1)

    levels = np.linspace(0, 5.4, 10)
    ax0 = fig.add_subplot(gs[0, 1:3])
    cs = ax0.contourf(xx, tt, h_test[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax0.set_ylabel('Time (h)')
    ax0.set_xticklabels([])
    ax0.text(0.05, 0.9, '{} Reference'.format(labels[0]), fontsize=18, transform=ax0.transAxes)

    ax1 = fig.add_subplot(gs[1, :2])
    cs = ax1.contourf(xx, tt, h_pred_2obs[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax1.set_ylabel('Time (h)')
    ax1.set_xticklabels([])
    ax1.text(0.05, 0.9, '{} PINN'.format(labels[1]), fontsize=18, transform=ax1.transAxes)

    ax2 = fig.add_subplot(gs[1, 2:]) 
    cs = ax2.contourf(xx, tt, h_interp_2obs[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.text(0.05, 0.9, '{} Interpolation'.format(labels[2]), fontsize=18, transform=ax2.transAxes)

    ax3 = fig.add_subplot(gs[2, :2])
    cs = ax3.contourf(xx, tt, h_pred_3obs[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax3.set_ylabel('Time (h)')
    ax3.set_xticklabels([])
    ax3.text(0.05, 0.9, '{} PINN'.format(labels[3]), fontsize=18, transform=ax3.transAxes)

    ax4 = fig.add_subplot(gs[2, 2:])
    cs = ax4.contourf(xx, tt, h_interp_3obs[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.text(0.05, 0.9, '{} Interpolation'.format(labels[4]), fontsize=18, transform=ax4.transAxes)

    ax5 = fig.add_subplot(gs[3, :2])
    cs = ax5.contourf(xx, tt, h_pred_4obs[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax5.set_ylabel('Time (h)')
    ax5.set_xticklabels([])
    ax5.text(0.05, 0.9, '{} PINN'.format(labels[5]), fontsize=18, transform=ax5.transAxes)
        
    ax6 = fig.add_subplot(gs[3, 2:])
    cs = ax6.contourf(xx, tt, h_interp_4obs[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax6.set_xticklabels([])
    ax6.set_yticklabels([])
    ax6.text(0.05, 0.9, '{} Interpolation'.format(labels[6]), fontsize=18, transform=ax6.transAxes)

    ax7 = fig.add_subplot(gs[4, :2])
    cs = ax7.contourf(xx, tt, h_pred_5obs[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax7.set_ylabel('Time (h)')
    ax7.set_xlabel('Distance upstream (m)')
    ax7.text(0.05, 0.9, '{} PINN'.format(labels[7]), fontsize=18, transform=ax7.transAxes)
        
    ax8 = fig.add_subplot(gs[4, 2:])
    cs = ax8.contourf(xx, tt, h_interp_5obs[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax8.set_xlabel('Distance upstream (m)')
    ax8.set_yticklabels([])
    ax8.text(0.05, 0.9, '{} Interpolation'.format(labels[8]), fontsize=18, transform=ax8.transAxes)

    cax = fig.add_axes([0.15, 0.05, 0.7, 0.01])
    cb = fig.colorbar(cs, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=16)
    cb.ax.yaxis.offsetText.set_fontsize(16)
    cb.set_label('Water depth (m)', fontsize=16)

    plt.savefig('figures/case6/contour.png',bbox_inches='tight')
    plt.close()


    tlist = [20, 50, 100, 200]
    plt.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots( 4, 4, figsize=(15, 12), sharex=True, sharey=True)
    for i in range(4):
        if i == 0:
            for k in range(len(tlist)): 
                axes[i,k].plot(x, h_test[tlist[k],:]+eles, 'ok', label='reference')
                axes[i,k].plot(x, h_pred_2obs[tlist[k],:]+eles, '-r', linewidth=3, label='PINN')
                axes[i,k].plot(x, h_interp_2obs[tlist[k],:]+eles, '-g', linewidth=3, label='interpolation')
                axes[i,k].fill_between(x.flatten(), eles, color='0.7')
                axes[i,k].set_title('t={} h'.format(int(tlist[k])), fontsize=18)   
                axes[i,k].set_ylim([0,5])
                axes[i,k].grid()
                if k == 0:
                    axes[i,k].legend(loc=2,prop={'size': 12.5})
        if i == 1:
            for k in range(len(tlist)):
                axes[i,k].plot(x, h_test[tlist[k],:]+eles, 'ok', label='reference')
                axes[i,k].plot(x, h_pred_3obs[tlist[k],:]+eles, '-r', linewidth=3, label='PINN')
                axes[i,k].plot(x, h_interp_3obs[tlist[k],:]+eles, '-g', linewidth=3, label='interpolation')
                axes[i,k].fill_between(x.flatten(), eles, color='0.7')
                axes[i,k].set_ylim([0,5])
                axes[i,k].grid()
        if i == 2:
            for k in range(len(tlist)):
                axes[i,k].plot(x, h_test[tlist[k],:]+eles, 'ok', label='reference')
                axes[i,k].plot(x, h_pred_4obs[tlist[k],:]+eles, '-r', linewidth=3, label='PINN')
                axes[i,k].plot(x, h_interp_4obs[tlist[k],:]+eles, '-g', linewidth=3, label='interpolation')
                axes[i,k].fill_between(x.flatten(), eles, color='0.7')
                axes[i,k].set_ylim([0,5])
                axes[i,k].grid() 
        if i == 3:
            for k in range(len(tlist)):
                axes[i,k].plot(x, h_test[tlist[k],:]+eles, 'ok', label='reference')
                axes[i,k].plot(x, h_pred_5obs[tlist[k],:]+eles, '-r', linewidth=3, label='PINN')
                axes[i,k].plot(x, h_interp_5obs[tlist[k],:]+eles, '-g', linewidth=3, label='interpolation')
                axes[i,k].fill_between(x.flatten(), eles, color='0.7')
                axes[i,k].set_ylim([0,5])
                axes[i,k].grid()
    for i in range(4):
        axes[i,0].set_ylabel('Water stage (m)')
    for k in range(len(tlist)):
        axes[3,k].set_xlabel('Distance upstream (m)')

    plt.tight_layout()
    plt.savefig('figures/case6/along_channel.png')
    plt.close()
