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
from scipy.optimize import Bounds, minimize, curve_fit,shgo, dual_annealing, brute, differential_evolution
#from customScipy import curve_fit
from matplotlib import cm
import matplotlib.patches as patches
import time
import random

import nlopt

#from ROS_LocAndForceEstimation import HerzianContactLoc
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

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
    
def compare_Z(shape,spacing,X,Y,Z_1,Z_2):
    fig = plt.figure(figsize=[2*(shape[0]+1),(shape[1]+1)])
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_xlim([0,spacing*(shape[0]+1)])
    ax1.set_ylim([0,spacing*(shape[1]+1)])
    """
    X = np.arange(0, (shape[0]+1)*spacing, 0.05)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.05)
    X, Y = np.meshgrid(X, Y)
    """
    plt.contourf(X,Y,Z_1,cmap = cm.coolwarm,extend='both',vmin=0,vmax=np.max(Z_1))
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
    
        ax1 = fig.add_subplot(1,2,1)
     
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_xlim([0,spacing*(shape[0]+1)])
    ax2.set_ylim([0,spacing*(shape[1]+1)])
    """
    X = np.arange(0, (shape[0]+1)*spacing, 0.05)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.05)
    X, Y = np.meshgrid(X, Y)
    """
    plt.contourf(X,Y,Z_2,cmap = cm.coolwarm,extend='both',vmin=0,vmax=np.max(Z_1))
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
    
    error = np.array([])
    Z_1f = Z_1.flatten()
    Z_2f = Z_2.flatten()
    for (z1,z2) in zip(Z_1f,Z_2f):
        error_i = (z1-z2)**2
        error = np.append(error,error_i)
        
    RMSE = (error.mean())**0.5
    print(RMSE)
    plt.show()
    return RMSE
    
def gaussian_r(r,p0,std):
    return p0*exp(-0.5*(r/std)**2)

def allInOne_alt(X,Y,p0,std,lx,ly,r_curve,theta,x0,y0):
    Z = np.zeros(X.shape)
    
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    X_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    Y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
    
    """
    if r_curve == 0:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
    elif r_curve > 0:
        r = Y_theta + r_curve
        alpha = np.divide(X_theta,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    else:
        r = -Y_theta- r_curve
        alpha = np.divide(X_theta,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    """
    if r_curve < -0.1:
        r_curve = 10/r_curve
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r + r_curve)
    elif r_curve > 0.1:
        r_curve = 10/r_curve
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,-(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r - r_curve)
    else: # between -0.1 and 0.1
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
        
    #X_new2 = np.reshape(X_new,(20,20))
    #Y_new2 = np.reshape(Y_new,(20,20))
      
    for i,(x,y) in enumerate(zip(X_new,Y_new)):
        if (abs(x) < lx/2 and abs(y) < ly/2):
            Z[i] = p0
        elif (abs(x) > lx/2 and abs(y) > ly/2):
            r = min((x-lx/2)**2+(y-ly/2)**2,
                    (x+lx/2)**2+(y-ly/2)**2,
                    (x-lx/2)**2+(y+ly/2)**2,
                    (x+lx/2)**2+(y+ly/2)**2)
            Z[i] = gaussian_r(sqrt(r),p0,std)
        elif x < -lx/2:
            Z[i] = gaussian_r(-lx/2-x,p0,std)
        elif x > lx/2:
            Z[i] = gaussian_r(x-lx/2,p0,std)
        elif y < -ly/2:
            Z[i] = gaussian_r(-ly/2-y,p0,std)
        elif y > ly/2:
            Z[i] = gaussian_r(y-ly/2,p0,std)
        else:
            Z[i] = p0
    
    """
    if r_curve == 0:
        X_new = X_bar.copy()
        Y_new = Y_bar.copy()
    elif r_curve > 0:
        r = Y_bar + r_curve
        alpha = np.divide(X_bar,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    else:
        r = -Y_bar - r_curve
        alpha = np.divide(X_bar,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
        
    X_f = X_new*cos(theta)+Y_new*sin(theta) + x0
    Y_f = -X_new*sin(theta)+Y_new*cos(theta) + y0
    """       
    return Z

def allInOne_alt_array(X,Y,p0,std,lx,ly,r_curve,theta,x0,y0):
    Z = np.zeros(X.shape)
    theta = theta/180*pi
    
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    X_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    Y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
    
    if r_curve < -0.1:
        r_curve = 10/r_curve
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r + r_curve)
    elif r_curve > 0.1:
        r_curve = 10/r_curve
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,-(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r - r_curve)
    else:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
    
    for i in range(X_new.shape[0]):
        for j in range(X_new.shape[1]):
            """
            x = X[i][j]
            y = Y[i][j]
            x_bar = x - x0
            y_bar = y - y0
            """
            x_theta = X_new[i][j]
            y_theta = Y_new[i][j]
            
            if (abs(x_theta) < lx/2 and abs(y_theta) < ly/2):
                Z[i][j] = p0
            elif (abs(x_theta) > lx/2 and abs(y_theta) > ly/2):
                r = min((x_theta-lx/2)**2+(y_theta-ly/2)**2,
                        (x_theta+lx/2)**2+(y_theta-ly/2)**2,
                        (x_theta-lx/2)**2+(y_theta+ly/2)**2,
                        (x_theta+lx/2)**2+(y_theta+ly/2)**2)
                Z[i][j] = gaussian_r(sqrt(r),p0,std)
            elif x_theta < -lx/2:
                Z[i][j] = gaussian_r(-lx/2-x_theta,p0,std)
            elif x_theta > lx/2:
                Z[i][j] = gaussian_r(x_theta-lx/2,p0,std)
            elif y_theta < -ly/2:
                Z[i][j] = gaussian_r(-ly/2-y_theta,p0,std)
            elif y_theta > ly/2:
                Z[i][j] = gaussian_r(y_theta-ly/2,p0,std)
            else:
                Z[i][j] = p0
    return Z


def allInOne_alt2(X,Y,p0,std,lx,ly,r_curve,theta,x0,y0):
    Z = np.zeros(X.shape)
    
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    X_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    Y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
    
    """
    if r_curve == 0:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
    elif r_curve > 0:
        r = Y_theta + r_curve
        alpha = np.divide(X_theta,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    else:
        r = -Y_theta- r_curve
        alpha = np.divide(X_theta,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    """
    if r_curve < -0.1:
        r_curve = -10/(10**(-r_curve-1))
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r + r_curve)
    elif r_curve > 0.1:
        r_curve = 10/(10**(r_curve-1))
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,-(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r - r_curve)
    else:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
        
    #X_new2 = np.reshape(X_new,(20,20))
    #Y_new2 = np.reshape(Y_new,(20,20))
      
    for i,(x,y) in enumerate(zip(X_new,Y_new)):
        if (abs(x) < lx/2 and abs(y) < ly/2):
            Z[i] = p0
        elif (abs(x) > lx/2 and abs(y) > ly/2):
            r = min((x-lx/2)**2+(y-ly/2)**2,
                    (x+lx/2)**2+(y-ly/2)**2,
                    (x-lx/2)**2+(y+ly/2)**2,
                    (x+lx/2)**2+(y+ly/2)**2)
            Z[i] = gaussian_r(sqrt(r),p0,std)
        elif x < -lx/2:
            Z[i] = gaussian_r(-lx/2-x,p0,std)
        elif x > lx/2:
            Z[i] = gaussian_r(x-lx/2,p0,std)
        elif y < -ly/2:
            Z[i] = gaussian_r(-ly/2-y,p0,std)
        elif y > ly/2:
            Z[i] = gaussian_r(y-ly/2,p0,std)
        else:
            Z[i] = p0
    
    """
    if r_curve == 0:
        X_new = X_bar.copy()
        Y_new = Y_bar.copy()
    elif r_curve > 0:
        r = Y_bar + r_curve
        alpha = np.divide(X_bar,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    else:
        r = -Y_bar - r_curve
        alpha = np.divide(X_bar,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
        
    X_f = X_new*cos(theta)+Y_new*sin(theta) + x0
    Y_f = -X_new*sin(theta)+Y_new*cos(theta) + y0
    """       
    return Z

def allInOne_alt_array2(X,Y,p0,std,lx,ly,r_curve,theta,x0,y0):
    Z = np.zeros(X.shape)
    theta = theta/180*pi
    
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    X_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    Y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
    
    if r_curve < -0.1:
        r_curve = -10/(10**(-r_curve-1))
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r + r_curve)
    elif r_curve > 0.1:
        r_curve = 10/(10**(r_curve-1))
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,-(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r - r_curve)
    else:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
    
    for i in range(X_new.shape[0]):
        for j in range(X_new.shape[1]):
            """
            x = X[i][j]
            y = Y[i][j]
            x_bar = x - x0
            y_bar = y - y0
            """
            x_theta = X_new[i][j]
            y_theta = Y_new[i][j]
            
            if (abs(x_theta) < lx/2 and abs(y_theta) < ly/2):
                Z[i][j] = p0
            elif (abs(x_theta) > lx/2 and abs(y_theta) > ly/2):
                r = min((x_theta-lx/2)**2+(y_theta-ly/2)**2,
                        (x_theta+lx/2)**2+(y_theta-ly/2)**2,
                        (x_theta-lx/2)**2+(y_theta+ly/2)**2,
                        (x_theta+lx/2)**2+(y_theta+ly/2)**2)
                Z[i][j] = gaussian_r(sqrt(r),p0,std)
            elif x_theta < -lx/2:
                Z[i][j] = gaussian_r(-lx/2-x_theta,p0,std)
            elif x_theta > lx/2:
                Z[i][j] = gaussian_r(x_theta-lx/2,p0,std)
            elif y_theta < -ly/2:
                Z[i][j] = gaussian_r(-ly/2-y_theta,p0,std)
            elif y_theta > ly/2:
                Z[i][j] = gaussian_r(y_theta-ly/2,p0,std)
            else:
                Z[i][j] = p0
    return Z

def getValue(X,Y,Z,xi,yi):
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    point = list(zip(xi,yi))
    #point = [(xi,yi)]
    zi = scipy.interpolate.griddata((X,Y),Z,point)
    zi = np.nan_to_num(zi)
    return zi

def goodGuess(array,shape,spacing):
    #normalize pressure
    array_0 = np.zeros(array.shape)
    #array_0.T[0] = array.T[0]
    array_0.T[0] = array.T[0].T/np.max(array.T[0])
    array_0.T[1] = array.T[1]
    array_0.T[2] = array.T[2]
    
    n0 = 0
    for i0 in range(len(array_0)):
        if array_0[i0][0] > 0.8:
            n0 += 1
    
    if n0 < 4:
        n0 = 4
    array_08 = array_0[:n0]
    #print(array_08)
    
    x0_x = np.mean(array_08.T[1])
    x0_y = np.mean(array_08.T[2])
    
    dist_array = []
    for i in range(n0):
        dist = sqrt((array_0[i][1]-x0_x)**2+(array_0[i][2]-x0_y)**2)
        dist_array.append(dist)
    
    dist_array = np.array(dist_array)            
    array_corners = np.zeros((4,3))
    for i in range(4):
        idx = np.argmax(dist_array)
        dist_array[idx] = -1
        array_corners[i] = array_08[idx]

    #print(array_corners)
    max_dist = 0
    idx_1 = -1
    idx_2 = -1
    for i in range(4):
        for j in range(i):
            dist = sqrt((array_corners[i][1]-array_corners[j][1])**2+(array_corners[i][2]-array_corners[j][1])**2)
            if dist > max_dist:
                max_dist = dist
                idx_1 = i
                idx_2 = j
      
    x0_lx = max_dist
        
    idxs = [0,1,2,3]
    idxs.remove(idx_1)
    idxs.remove(idx_2)
    x0_ly =sqrt((array_corners[idxs[0]][0]-array_corners[idxs[1]][1])**2+(array_corners[idxs[0]][2]-array_corners[idxs[1]][1])**2)
    
    x0 = [array[0][0], 3, x0_lx, x0_ly, 0, 0, x0_x, x0_y]
    print(x0)
    lb = [array[0][0], 0, 0, 0, -100, -90, min(array_08.T[1]), min(array_08.T[2])] #[x0[0], 0, 0, 0, -100, -90, min(array_n.T[1][nx[0]],array_n.T[1][nx[1]]), min(array_n.T[2][ny[0]],array_n.T[2][ny[1]])] #[x0[0], 0, 0, 0, 0.1, 0.1, 0, -20, 1, -90, min(array_n.T[1][nx[0]],array_n.T[1][nx[1]]), min(array_n.T[2][ny[0]],array_n.T[2][ny[1]])]
    ub = [10*array[0][0], 10, spacing*max(shape), spacing*max(shape), 100, 90, max(array_08.T[1]), max(array_08.T[2])]
    return x0, lb, ub

def RandomParams(shape,spacing):
    """
    lb = [x0[0], 0, -90, 0, 0, min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    ub = [x0[0]*10, 10, 90, spacing*max(shape), spacing*max(shape), max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    allInOne(X,Y,p0,std,lx,ly,S_x,S_y,S,r_curve,F,theta,x0,y0):
    """
    p0 = random.uniform(10000,100000)
    std = random.uniform(0,10)
    lx = random.uniform(spacing,spacing*max(shape)/2)
    ly = random.uniform(spacing,lx)
    #S_x = random.uniform(1,10)
    #S_y = random.uniform(1,10)
    #S = random.uniform(0,1)
    r_curve = random.uniform(-3,3)
    #F = random.uniform(1,1)
    theta = random.uniform(-90,90)
    #x0 = random.uniform(spacing,spacing*(shape[0]-1))
    #y0 = random.uniform(spacing,spacing*(shape[1]-1))
    x0 = random.uniform(lx,spacing*shape[0]-lx)
    y0 = random.uniform(lx,spacing*shape[1]-lx)
    return [p0,std,lx,ly,r_curve,theta,x0,y0]#[p0,std,lx,ly,S_x,S_y,S,r_curve,F,theta,x0,y0]

def PressureDistribution_global(params,array):
    #params = [p0,std,lx,ly,r_curve,F,theta,x0,y0]
    X_i = array.T[1]
    Y_i = array.T[2]
    Z_meas = array.T[0]
    Z_i = allInOne_alt(X_i,Y_i,*params)
    #plot_Z([5,9],4.5,X_new,Y_new,Z_new)
    #print("time AllInOne: "+str(t1-t0))
    #print("time getValue: "+str(t2-t1))
    #print(params)
    Z_error = abs(Z_i-Z_meas)
    return np.sum(Z_error)
    

def Optimization(xi,yi,pressureArray, shape, spacing, t=6,n=9):
    """
    n: amount of points used
    """
    global x0
    n_p = n
    
    # t0 = time.time()
    amount = int(shape[0]*shape[1])
    
    array = np.zeros((amount,3)) # n points with highest pressure; 2: pressure; 3: x coordinate; 4: y coordinate
    
    for i in range(amount):
        idx = np.argmax(pressureArray)
        array[i][0] = pressureArray[idx]    
        array[i][1] = xi[idx]
        array[i][2] = yi[idx]
        
        pressureArray[idx] = -10**9
    #print(array)
    
    #x0 = [array[0][1],2,array[0][2],array[0][3]] # [k, a, x_c, y_c]
    #bnds = scipy.optimize.Bounds([array[0][1],0,array[0][2]-spacing,array[0][3]-spacing],[float('inf'),float('inf'),array[0][2]-spacing,array[0][3]-spacing])
    """
    if not success:
        x0 = [array[0][1],1,array[0][2],array[0][3]] # [k, x_c, y_c]
    """
    #x0 = [array[0][1]/gaussian(0,0,1,1,0,0),1,array[0][2],array[0][3]]
    #bnds = scipy.optimize.Bounds([array[0][1]//gaussian(0,0,1,1,0,0),0,min(array[0][2]-spacing,0),min(array[0][3]-spacing,0)],[float('inf'),float('inf'),max(array[0][2]+spacing,shape[0]*spacing),max(array[0][3]+spacing,shape[0]*spacing)])
    
    #print(array)
    #x0, lb, ub = goodGuess(array[:n_p],shape,spacing)
    
    ### normalize
    length_0 = max(shape[0]*spacing,shape[1]*spacing)
    array_0 = np.zeros(array.shape)
    #array_0.T[0] = array.T[0]
    array_0.T[0] = array.T[0].T/np.max(array.T[0])
    array_0.T[1] = array.T[1]/(length_0)
    array_0.T[2] = array.T[2]/(length_0)
    
    n0 = 0
    for i0 in range(len(array_0)):
        if array_0[i0][0] > 0.5:
            n0 += 1
    
    array_0 = array_0[:n0]
    #print(array_0)
    l1 = max(array_0.T[1]) - min(array_0.T[1])
    l2 = max(array_0.T[2]) - min(array_0.T[2])
    lx_0 = max(l1,l2)
    ly_0 = min(l1,l2)
    
    
    #print(n0)
    diff_x = False
    diff_y = False
    
    array_n = array
    
    n = 3
    nx = []
    ny = []
    while not (diff_x and diff_y):
        for i in range(n):
            for j in range(i):
                if (array[i][1] != array[j][1]) and (not diff_x):
                    diff_x = True
                    nx.append(j)
                    nx.append(i)
                if (array[i][2] != array[j][2]) and (not diff_y):
                    diff_y = True
                    ny.append(j)
                    ny.append(i)
        if not (diff_x and diff_y):
            n += 1

    x0 = [array_n[0][0],3,lx_0,ly_0,0,0,array_n.T[1][0],array_n.T[2][0]]
    
    """
    #print(n0)
    diff_x = False
    diff_y = False
    
    array_n = array
    
    n = 3
    nx = []
    ny = []
    while not (diff_x and diff_y):
        for i in range(n):
            for j in range(i):
                if (array[i][1] != array[j][1]) and (not diff_x):
                    diff_x = True
                    nx.append(j)
                    nx.append(i)
                if (array[i][2] != array[j][2]) and (not diff_y):
                    diff_y = True
                    ny.append(j)
                    ny.append(i)
        if not (diff_x and diff_y):
            n += 1
    
    # x0 = ['p0','std','lx','ly','r_curve','F','theta','x0','y0']
    
    if n0 >= 5:
        #x0 = [array_n[0][0],2,2,2,0,1,0,(array_n.T[1][nx[0]]+array_n.T[1][nx[1]])/2,(array_n.T[2][ny[0]]+array_n.T[2][ny[1]])/2]
        x0 = [array_n[0][0],5,2,2,0,1,0,(array_n.T[1][nx[0]]),(array_n.T[2][ny[0]])]
    else:
        #x0 = [array_n[0][0],3,0,0,0,1,0,(array_n.T[1][nx[0]]+array_n.T[1][nx[1]])/2,(array_n.T[2][ny[0]]+array_n.T[2][ny[1]])/2]
        x0 = [array_n[0][0],3,0,0,0,1,0,(array_n.T[1][nx[0]]),(array_n.T[2][ny[0]])]
     #x0 = [array_n[0][1],0.1,array_n[0][2],array_n[0][3]]
    
    #print(x0)
    """
    """
    bnds = scipy.optimize.Bounds([x0[0],0.01,-90.0,0.0,0.0,min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]),min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])],
                                  [float('inf'),float('inf'),90,float('inf'),float('inf'),max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]),max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])])
    
    """
    
    lb = [x0[0], 0.0001, 0, 0, 0, -90, min(array_n.T[1][nx[0]],array_n.T[1][nx[1]]), min(array_n.T[2][ny[0]],array_n.T[2][ny[1]])] #[x0[0], 0, 0, 0, 0.1, 0.1, 0, -20, 1, -90, min(array_n.T[1][nx[0]],array_n.T[1][nx[1]]), min(array_n.T[2][ny[0]],array_n.T[2][ny[1]])]
    ub = [x0[0]*10, 10,spacing*max(shape), spacing*max(shape), 3, 90, max(array_n.T[1][nx[0]],array_n.T[1][nx[1]]), max(array_n.T[2][ny[0]],array_n.T[2][ny[1]])] #[x0[0]*10, 10,spacing*max(shape), spacing*max(shape), 20, 20, 1, 20, 3, 90, max(array_n.T[1][nx[0]],array_n.T[1][nx[1]]), max(array_n.T[2][ny[0]],array_n.T[2][ny[1]])]
    
    #x_scale = [x0[0],5,spacing*max(shape), spacing*max(shape),10,90,spacing*max(shape),spacing*max(shape)]
    #time0 = time.time()
    #E = minimize(ObjFun,x0,args=(array_n,n_p),method = meth[2],bounds=bnds,options=opt_2)
    
    
    xdata = array.T[1:].T[:n_p].T
    xdata = np.append(xdata,np.zeros((1,xdata.shape[1])),axis=0)
    ydata = array.T[0][:n_p]
    
    #print(xdata)
    #print(ydata)
    meth = ['trf','dogbox']
    bounds = []
    for (lbi,ubi) in zip(lb,ub):
        bounds.append((lbi,ubi,(ubi-lbi)/5)) #brute
        #bounds.append((lbi,ubi))
    #print(bounds)
    
    rranges = tuple(bounds)
    #print(rranges)
    
    #E = curve_fit(PressureDistribution, xdata, ydata, x0, bounds=(lb, ub), method = meth[0], maxfev=100,verbose=0,ftol=1e-4)#,diff_step=1,x_scale=x_scale)#,diff_step=100)
    
    #E = minimize(PressureDistribution_global,x0,args=(array),method='SLSQP')
    E = brute(PressureDistribution_global,rranges,args=(array,),finish=minimize,workers=-1)
    #E = differential_evolution(PressureDistribution_global,bounds,args=(array,),workers=-1)
    #E = shgo(PressureDistribution_global,bounds,args=(array,))
    #E = dual_annealing(PressureDistribution_global,bounds,args=(array,),x0=x0)
    
    #print(E)
    ### denormalize
    #E.x[0] = E.x[0]*np.max(array.T[1])
    #E.x[4] = E.x[4]*length_n
    #E.x[5] = E.x[5]*length_n
    """
    if E.success:
        x0 = E.x
    """
    #E.x[0] = E.x[0]/(E.x[1])**2 #==> E.x = [p0,z,x_c,y_c] p0 in Pa
    #print(E)
    #print(1/(time.time()-t0))
    return array,E

if __name__ == '__main__':
    
    shape = [5,9]
    spacing = 4.5
    
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
    times = np.array([])
            
    for ind in range(100):#len(list_files)):
        print("------------------------------------------------")
        print(ind+1)
    
        params_names = ['p0','std','lx','ly','r_curve','theta','x0','y0'] #['p0','std','lx','ly','S_x','S_y','S','r_curve','F','theta','x0','y0']
        params = RandomParams(shape,spacing)
        
        #params = [10,2,0,10,0.5,13.5,22.5]
        #params = [10,1,20,0.5,40,0,13.5,22.5]
        #params = [10,1,15,2,2,1,0.1,-7,2.1,45,13.5,22.5]
        print(dict(zip(params_names,params)))
        
        X = np.linspace(0, (shape[0]+1)*spacing, 50)
        Y = np.linspace(0, (shape[1]+1)*spacing, 50)
        X, Y = np.meshgrid(X, Y)
        
        """
        #Z = roundedSurface_array(X,Y,params[0],params[1],params[2],params[3],params[4],params[5],params[6])
        X_0,Y_0,Z_0 = Surface(X,Y,10,1,15,2)
        plot_Z(shape,spacing,X_0,Y_0,Z_0)
        
        X_1,Y_1,Z_1 = Scaling(X_0,Y_0,Z_0,1,10,0.1)
        plot_Z(shape,spacing,X_1,Y_1,Z_1)
        
        X_2,Y_2,Z_2 = Curve(X_1,Y_1,Z_1,0,0)
        plot_Z(shape,spacing,X_2,Y_2,Z_2,Z_0)
                
        X_3,Y_3,Z_3 = RotateAndTranslate(X_2,Y_2,Z_2,0,13.5,22.5)
        plot_Z(shape,spacing,X_3,Y_3,Z_3,Z_3)
        #plot pressure distribitution
        #plot_roundedSurface(shape,spacing,params)
        """
        #X_f,Y_f,Z_f = allInOne(X,Y,10,1,15,2,2,1,0.1,5,2,45,13.5,22.5)
        Z_1 = allInOne_alt_array2(X,Y,*params)
        #X_f,Y_f,Z_f = allInOne_MeshUnknown(*params)
        #plot_Z(shape,spacing,X,Y,Z_1,Z_1)
        Z_i = allInOne_alt2(xi,yi,*params)
        
        t0 = time.time()
        
        #results = np.zeros((4,1))
        array, E = Optimization(xi,yi,Z_i.copy(),shape,spacing,n=30)
        dur = time.time()-t0
        print("time to execute: "+str(dur))
        print(dict(zip(params_names,E)))
        print('parameter error percentage:')
        error_params = (params-E)/E*100
        print(dict(zip(params_names,error_params)))
        Z_2 = allInOne_alt_array2(X,Y,*E)#.x)
        RMSE_i = compare_Z(shape,spacing,X,Y,Z_1,Z_2)
        
        RMSE = np.append(RMSE,RMSE_i)
        times = np.append(times,dur)
        
    print("mean error: "+str(RMSE.mean()))
    print("max error: "+str(max(RMSE)))
    print("mean times: "+str(times.mean()))
    print("max time: "+str(max(times)))
        
        