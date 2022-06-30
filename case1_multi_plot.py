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
import pickle
import os
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from SVE_module_dynamic import SVE

import pdb


def findNearset(xx, xb):
    
    dist = np.abs(xx-xb)
    ind = np.argwhere(dist==dist.min())[0][0]
    if xx[ind] > xb:
        return ind-1
    else:
        return ind

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


def fh(x, t, u, n):
    return np.power(  (-7/3)*(n*n*u*u*(x-u*t)) , 3/7 )


if __name__ == "__main__": 
    
    n = 0.005
    u = 1.0
    
    layers = [2] + 3*[1*32] + [1]
    
    dt = 30
    dx = 30
    t = np.arange(0, 3600+dt, dt)
    x = np.arange(0, 3600+dx, dx)

    Nt = t.shape[0]
    Nx = x.shape[0]
    h_exact = np.zeros([Nt, Nx])
    for i in range(Nt):
        xb = u*t[i]
        indb = findNearset(x, xb)
        h_exact[i,:indb+1] = fh(x[:indb+1], t[i], u, n)

    X, T = np.meshgrid(x,t)
    
    x_star = []
    t_star = []
    h_star = []
    for i in range(Nt):
        xb = u*t[i]
        indb = findNearset(x, xb) 
        x_star += x[0:indb+1].tolist()
        t_star += [i*dt]*len(x[0:indb+1])
        h_star += h_exact[i,0:indb+1].tolist()
    x_star = np.array(x_star)
    t_star = np.array(t_star)
    X_star = np.hstack((x_star[:,None], t_star[:,None]))
    h_star = np.array(h_star)[:,None]
    
    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    ## IC
    tsteps = [0] + [Nt-1]
    for i, tstep in enumerate(tsteps):
        if i == 0:
            xx1_ = np.hstack((X[tstep:tstep+1,:].T, T[tstep:tstep+1,:].T))[0].reshape([1,2])
            hh1_ = add_noise(h_exact[tstep:tstep+1,:].T[0].reshape([1,1]))
            xx1 = xx1_
            hh1 = hh1_
        else:
            xx1_ = np.hstack((X[tstep:tstep+1,:].T, T[tstep:tstep+1,:].T))
            hh1_ = add_noise(h_exact[tstep:tstep+1,:]).T
            xx1 = np.vstack((xx1, xx1_))
            hh1 = np.vstack((hh1, hh1_))

    ## dnstrm BC
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))   ## dnstrm BC
    hh2 = h_exact[:,0:1]
    ## moving BC: h(ut,t)=0
    x_dnstrm = []
    hh3 = []
    for i in range(Nt):
        xb = u*t[i]
        indb = findNearset(x, xb)
        x_dnstrm.append(x[indb])
        hh3.append(h_exact[i, indb])
    x_dnstrm = np.array(x_dnstrm)[:,None]
    t_dnstrm = T[:,0:1]
    xx3 = np.hstack((x_dnstrm, t_dnstrm))
    hh3 = np.array(hh3)[:,None]

    X_IC = xx1
    h_IC = hh1
    X_BC = np.vstack([xx2, xx3])
    h_BC = np.vstack([hh2, hh3])

    useObs = False
    ## obs 
    ind_obs = [int(30*60./dx)]  # mid-point
    t_obs = np.array([])
    x_obs = np.array([])
    h_obs = np.array([])
    for iobs in ind_obs:
        indt = int(x[iobs]/u/dt)
        t_obs = np.append( t_obs, t[indt:] )
        x_obs = np.append( x_obs, x[iobs]*np.ones(len(t[indt:])) )
        h_obs = np.append( h_obs, add_noise(h_exact[indt:, iobs]) )
    X_obs = np.vstack([x_obs, t_obs]).T
    h_obs = h_obs[:,None]

    
    X_f_train = X_star
    slope = 0

    exist_mode = 2
    saved_path = 'saved_model/case1/PINN_SVE.pickle'
    weight_path = 'saved_model/case1/weights.out'
    # Training
    model2 = SVE(X_IC,
                X_BC,
                X_obs,
                X_f_train,
                h_IC,
                h_BC,
                h_obs,
                layers,
                lb, ub, slope, n, u,
                X_star, h_star,
                ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path,
                useObs=True)
    
    # Test data
    Nt_test = Nt
    N_test = Nt_test * Nx    ## Nt_test x Nx
    X_test = X_star[:N_test,:]
    x_test = X_test[:,0:1]
    t_test = X_test[:,1:2]
    h_test = h_star[:N_test,:]
    

    h_pred2 = model2.predict(x_test, t_test)
    error_h2 = np.linalg.norm(h_test-h_pred2,2)/np.linalg.norm(h_test,2)
    print('Error2 h: %e' % (error_h2))

    rmse_h = np.sqrt(((h_test - h_pred2) ** 2).mean())
    print('RMSE h: %.3f m' % rmse_h )
    
    # reshape
    h_pred_re2 = np.zeros([Nt_test, Nx])
    h_test_re = np.zeros([Nt_test, Nx])
    counter = 0
    for i in range(Nt):
        xb = u*t[i]
        indb = findNearset(x, xb)
        h_pred_re2[i,0:indb+1] = h_pred2.flatten()[counter:counter+indb+1]
        h_test_re[i,0:indb+1] = h_test.flatten()[counter:counter+indb+1]
        counter += indb+1

    #h_pred1 = h_pred_re1
    h_pred2 = h_pred_re2
    h_test = h_test_re
    
    xx, tt = np.meshgrid(x.flatten(), t[:Nt_test])
    plt.rcParams.update({'font.size': 16})
    labels = ['(a)', '(b)', '(c)']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.25), sharey=True)
    axes = axes.ravel()
    levels = np.linspace(0, 0.6, 9)
    cs = axes[0].contourf(xx, tt, h_test[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Time (s)')
    axes[0].set_title('Reference')

    cs = axes[1].contourf(xx, tt, h_pred2[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_title('Prediction')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(cs, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=14)
    cb.ax.yaxis.offsetText.set_fontsize(14)
    cb.set_label('Water depth (m)', fontsize=14)

    for i in range(len(axes)):
        axes[i].text(0.05, 0.9, '{}'.format(labels[i]), fontsize=16, transform=axes[i].transAxes)

    plt.tight_layout() 
    plt.savefig('figures/case1/contour.png')
    plt.close()

    for i in range(Nt):
        xb = u*t[i]
        indb = findNearset(x, xb)
        h_test[i, indb+1:] = None
        h_pred2[i, indb+1:] = None
    tlist = np.array([20, 30, 40, 60]) * (60./dt)
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots( 2, 2, figsize=(15, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    for k in range(len(tlist)):
        axes[k].plot(x, h_test[int(tlist[k]),:], 'ok', label='reference')
        axes[k].plot(x, h_pred2[int(tlist[k]),:], '-r', linewidth=2, label='PINN')
        axes[k].text(0.85, 0.85, 't={} s'.format(int(tlist[k]*dt)), fontsize=15, transform=axes[k].transAxes)
        if k in [0,2]:
            axes[k].set_ylabel('Water stage (m)')
        axes[k].set_ylim([0,0.6])
        axes[k].grid()
        if k in [2,3]:
            axes[k].set_xlabel('Distance (m)')
        if k == 0:
            axes[k].legend(loc=4,prop={'size': 14})

    plt.tight_layout()
    plt.savefig('figures/case1/along_channel.png')
    plt.close()
