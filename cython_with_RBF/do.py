#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 15:09:43 2022

@author: pc-robotiq
"""
import os
import numpy as np
import pandas as pd
import scipy
import math
from math import sqrt,pi,exp,sin,cos
from scipy.optimize import Bounds, minimize, curve_fit
from scipy.interpolate import RBFInterpolator, Rbf
#from customScipy import curve_fit
from matplotlib import cm
import matplotlib.patches as patches
import time
import random

#from ROS_LocAndForceEstimation import HerzianContactLoc
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

from PressureReconstruction import Optimization, calc_Z

def plot_Z(shape,spacing,X,Y,Z,Z_max=np.array([])):
    if not Z_max.any():
        Z_max = Z
    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    """
    X = np.arange(0, (shape[0]+1)*spacing, 0.05)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.05)
    X, Y = np.meshgrid(X, Y)
    """
    plt.contourf(X,Y,Z,cmap = cm.coolwarm,extend='both',vmin=0,vmax=np.max(Z_max))
    plt.colorbar()
    
    num_cir = shape[0]*shape[1]
    for i in range(num_cir):
        x = spacing+spacing*(i % shape[0])
        y = spacing+spacing*(i % shape[1])
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
    
    plt.show()
    
def compare_Z_with_rbf(shape,spacing,X,Y,Z_1,Z_2,Z_rbf,plot=True):
    if plot:
        fig = plt.figure(figsize=[2*(shape[0]+1),(shape[1]+1)])
        ax1 = fig.add_subplot(1,3,1)
        ax1.set_xlim([0,spacing*(shape[0]+1)])
        ax1.set_ylim([0,spacing*(shape[1]+1)])
        ax1.set_title('True pressure distribution')
        """
        X = np.arange(0, (shape[0]+1)*spacing, 0.05)
        Y = np.arange(0, (shape[1]+1)*spacing, 0.05)
        X, Y = np.meshgrid(X, Y)
        """
        levels = np.linspace(0.1*np.max(Z_1),0.9*np.max(Z_1),6)
        plt.contourf(X,Y,Z_1,levels,cmap = cm.coolwarm,extend='both',vmin=0,vmax=np.max(Z_1))
        plt.colorbar()
        
        num_cir = shape[0]*shape[1]
        for i in range(num_cir):
            x = spacing+spacing*(i % shape[0])
            y = spacing+spacing*(i % shape[1])
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            ax1.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
        
        ax2 = fig.add_subplot(1,3,2)
        ax2.set_xlim([0,spacing*(shape[0]+1)])
        ax2.set_ylim([0,spacing*(shape[1]+1)])
        ax2.set_title('RBF pressure distribution')
        """
        X = np.arange(0, (shape[0]+1)*spacing, 0.05)
        Y = np.arange(0, (shape[1]+1)*spacing, 0.05)
        X, Y = np.meshgrid(X, Y)
        """
        plt.contourf(X,Y,Z_rbf,levels,cmap = cm.coolwarm,extend='both',vmin=0,vmax=np.max(Z_1))
        plt.colorbar()
        
        num_cir = shape[0]*shape[1]
        for i in range(num_cir):
            x = spacing+spacing*(i % shape[0])
            y = spacing+spacing*(i % shape[1])
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            ax2.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
         
        ax3 = fig.add_subplot(1,3,3)
        ax3.set_xlim([0,spacing*(shape[0]+1)])
        ax3.set_ylim([0,spacing*(shape[1]+1)])
        ax3.set_title('Reconstructed pressure distribution')
        """
        X = np.arange(0, (shape[0]+1)*spacing, 0.05)
        Y = np.arange(0, (shape[1]+1)*spacing, 0.05)
        X, Y = np.meshgrid(X, Y)
        """
        plt.contourf(X,Y,Z_2,levels,cmap = cm.coolwarm,extend='both',vmin=0,vmax=np.max(Z_1))
        plt.colorbar()
        
        num_cir = shape[0]*shape[1]
        for i in range(num_cir):
            x = spacing+spacing*(i % shape[0])
            y = spacing+spacing*(i % shape[1])
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            ax3.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
        
    error = np.array([])
    Z_1f = Z_1.flatten()
    Z_2f = Z_2.flatten()
    for (z1,z2) in zip(Z_1f,Z_2f):
        error_i = (z1-z2)**2
        error = np.append(error,error_i)
        
    RMSE = (error.mean())**0.5/np.max(Z_1)*100
    #print('error pressure distribution: '+str(RMSE))
    
    error_rbf = np.array([])
    Z_rbff = Z_rbf.flatten()
    for (z1,zrbf) in zip(Z_1f,Z_rbff):
        error_j = (z1-zrbf)**2
        error_rbf = np.append(error_rbf,error_j)
        
    RMSE_rbf = (error_rbf.mean())**0.5/np.max(Z_1)*100
    #print('error rbf pressure distribution: '+str(RMSE_rbf))
    plt.show()
    return RMSE, RMSE_rbf

def RandomParams(shape,spacing):
    """
    lb = [x0[0], 0, -90, 0, 0, min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    ub = [x0[0]*10, 10, 90, spacing*max(shape), spacing*max(shape), max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    allInOne(X,Y,p0,std,lx,ly,S_x,S_y,S,r_curve,F,theta,x0,y0):
    """
    p0 = random.uniform(10,100)
    std = random.uniform(1,10)
    lx = random.uniform(spacing,spacing*max(shape)/2)
    ly = random.uniform(0,lx)
    #S_x = random.uniform(1,10)
    #S_y = random.uniform(1,10)
    #S = random.uniform(0,1)
    r_curve = random.uniform(0,3)
    #F = random.uniform(1,1)
    theta = random.uniform(-90,90)
    x0 = random.uniform(spacing,spacing*(shape[0]-1))
    y0 = random.uniform(spacing,spacing*(shape[1]-1))
    return [p0,std,lx,ly,r_curve,theta,x0,y0]#[p0,std,lx,ly,S_x,S_y,S,r_curve,F,theta,x0,y0]

def getValue(X,Y,Z,xi,yi):
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    point = list(zip(xi,yi))
    #point = [(xi,yi)]
    zi = scipy.interpolate.griddata((X,Y),Z,point)
    zi = np.nan_to_num(zi)
    return zi


if __name__ == '__main__':
    
    shape = [5,9]
    spacing = 4.5
    
    n = 50
    m = 10 # keep between 10 - 15
    p_m = 0.75
    max_it = 5
    max_t = 45
    
    list_time = []
    xi = []
    yi = []
    
    for x_i in range(shape[0]):
        for y_i in range(shape[1]):
            xi.append(spacing+x_i*spacing)
            yi.append(spacing+y_i*spacing)
            
    xi = np.array(xi)
    yi = np.array(yi)
            
    RMSE = np.array([])
    RMSE_rbf = np.array([])
    times = np.array([])
    
    params_names = ['p0','std','lx','ly','r_curve','theta','x0','y0']
    X = np.linspace(0, (shape[0]+1)*spacing, n)
    Y = np.linspace(0, (shape[1]+1)*spacing, n)
    X, Y = np.meshgrid(X, Y)
    
    X_rbf = np.linspace(0, (shape[0]+1)*spacing, m)
    Y_rbf = np.linspace(0, (shape[1]+1)*spacing, m)
    X_rbf, Y_rbf = np.meshgrid(X_rbf, Y_rbf)
    
    list_x0 = np.zeros((max_it,8))
        
    for ind in range(100):#len(list_files)):
        print("------------------------------------------------")
        print(ind+1)
        
        Z_all_real = np.zeros((n,n))
        Z_all_recon = np.zeros((n,n))
        #['p0','std','lx','ly','S_x','S_y','S','r_curve','F','theta','x0','y0']
        t0 = time.time()
        for i in range(5):
            params = RandomParams(shape,spacing)
            """
            if i == 0:
                params = [10, 1, 12, 0.5, 0, 0, 17.5, 22.5]
                #params = [10, 2, 0, 0, 0, 0, 13.5, 22.5]
                #params = [10,2,7,0,1.5,-45,5,22.5]
            elif i == 1:
                params = [10, 1, 12, 0.5, 0, 90, 4.5, 35]
            elif i == 2:
                params = [10, 1, 8, 0.5, 1.4, -45, 5, 22.5]
            """
            #params = [50855, 8.35, 17.9,9.5,0.259,65.95,9.11,22.23]
            
            #print(dict(zip(params_names,params)))
            
            #X_f,Y_f,Z_f = allInOne(X,Y,10,1,15,2,2,1,0.1,5,2,45,13.5,22.5)
            Z_1 = calc_Z(X.flatten(),Y.flatten(),*params)
            Z_1 = np.reshape(Z_1,X.shape)
            Z_all_real = Z_all_real + Z_1
            #X_f,Y_f,Z_f = allInOne_MeshUnknown(*params)
            #plot_Z(shape,spacing,X,Y,Z_1,Z_1)
            #Z_i = calc_Z(xi,yi,*params)
            
            #print(list(zip(xi,yi,Z_i)))
            #print(Z_i)
        
        
        #results = np.zeros((4,1))
        t00 = time.time()
        Z_i = getValue(X, Y, Z_all_real, xi, yi)
        t1 = time.time()
        rbfi = Rbf(xi,yi,Z_i,function='gaussian') #always +-2.5 ms
        t2 = time.time()
        Z_rbf = rbfi(X_rbf,Y_rbf)
        t3 = time.time()
        array_rbf = np.array([X_rbf.flatten(),Y_rbf.flatten(),Z_rbf.flatten()]).T
        
        idxs = []
        for idx,el in enumerate(array_rbf):
            if el[0] < spacing or el[0] > shape[0]*spacing:
                idxs.append(idx)
            elif el[1] < spacing or el[1] > shape[1]*spacing:
                idxs.append(idx)
        array_rbf = np.delete(array_rbf,idxs,axis=0).T
        toi = time.time()
        array, list_E = Optimization(xi,yi,Z_i.copy(),shape,spacing, array_rbf, list_x0, n=round(p_m*array_rbf.shape[1]),it_max=max_it,t_max=max_t)
        toe = time.time()
        print("time opt: "+str((toe-toi)*10**3))
        """
        print(dict(zip(params_names,E.x)))
        print('parameter error percentage:')
        error_params = (params-E.x)/E.x*100
        print(dict(zip(params_names,error_params)))
        """
        dur = (time.time()-t1)*10**3
        print("total time to execute: "+str(dur))
        #print("time not opt: "+str((dur-(toe-toi))*10**3))
        
        for E in list_E:
            Z_2 = calc_Z(X.flatten(),Y.flatten(),*E.x)
            Z_2 = np.reshape(Z_2,X.shape)
            Z_all_recon = Z_all_recon + Z_2
        #print('t1 - t0: '+str(t1-t0))
        #print('t2 - t1: '+str(t2-t1))
        #print('t3 - t2: '+str(t3-t2))
        Z_all_rbf = getValue(X_rbf, Y_rbf, Z_rbf, X.flatten(), Y.flatten())
        Z_all_rbf = Z_all_rbf.reshape(X.shape)
        
        RMSE_i,RMSE_j = compare_Z_with_rbf(shape,spacing,X,Y,Z_all_real,Z_all_recon,Z_all_rbf,plot=False)

        RMSE = np.append(RMSE,RMSE_i)
        RMSE_rbf = np.append(RMSE_rbf,RMSE_j)
        times = np.append(times,dur)
    
    print("###############################################")    
    print("mean error: "+str(RMSE.mean()))
    print("mean error rbf only: "+str(RMSE_rbf.mean()))
    print("max error: "+str(max(RMSE)))
    print("mean times: "+str(times.mean()))
    print("max time: "+str(max(times)))
    
    fig = plt.figure()
    plt.plot(times,'o')
        