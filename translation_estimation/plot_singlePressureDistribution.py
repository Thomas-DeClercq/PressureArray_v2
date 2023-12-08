#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:44:52 2022

@author: pc-robotiq
"""
import random, math
import numpy as np
import pandas as pd
from math import sin, cos, pi
import time

from matplotlib import pyplot as plt
from matplotlib import cm, animation
import matplotlib.patches as patches

#from PressureReconstruction import calc_Z
import sys
sys.path.insert(1,'/home/thomas/pythonScripts/PressureArray_v2/PR_cython')
from PressureReconstruction_update210623 import calc_Z, Optimization

#plt.rcParams.update({"font.family":"serif"})
plt.rcParams['font.serif'] = ['Times New Roman']

def plot_params(shape,spacing,X,Y,params):
    Z_n = np.zeros(X.shape)
    for i in range(params.shape[0]):
        Z_1 = calc_Z(X.flatten(),Y.flatten(),*params_i)
        Z_1 = np.reshape(Z_1,X.shape)
        Z_n = Z_n + Z_1
        
    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    """
    X = np.arange(0, (shape[0]+1)*spacing, 0.05)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.05)
    X, Y = np.meshgrid(X, Y)
    """
    plt.contourf(X,Y,Z_n,cmap = cm.coolwarm,extend='both')
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

def RandomParams(shape,spacing):
    """
    lb = [x0[0], 0, -90, 0, 0, min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    ub = [x0[0]*10, 10, 90, spacing*max(shape), spacing*max(shape), max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    allInOne(X,Y,p0,std,lx,ly,S_x,S_y,S,r_curve,F,theta,x0,y0):
    """
    p0 = random.uniform(1000,300000)
    std = random.uniform(1,10)
    lx = random.uniform(0,spacing*max(shape)/2)
    ly = random.uniform(0,spacing*max(shape)/2)
    #S_x = random.uniform(1,10)
    #S_y = random.uniform(1,10)
    #S = random.uniform(0,1)
    r_curve = random.uniform(0,4)
    #F = random.uniform(1,1)
    theta = random.uniform(-180,180)
    x0 = random.uniform(spacing,spacing*(shape[0]))
    y0 = random.uniform(spacing,spacing*(shape[1]))
    return [p0,std,lx,ly,r_curve,theta,x0,y0]#[p0,std,lx,ly,S_x,S_y,S,r_curve,F,theta,x0,y0]

if __name__ == "__main__":
    
    shape = [4,8]
    spacing = 4.5
    
    # p0, sigma, lx, ly, r, alpha, x0, y0 
    params = [10,2,10,3,1,0,11.25,22.5]

    n = 50            
    X = np.linspace(0, (shape[0]+1)*spacing, n)
    Y = np.linspace(0, (shape[1]+1)*spacing, n)
    X, Y = np.meshgrid(X, Y)
    

    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    ax.set_xticklabels(['0', '2.5','5','7.5','10','12.5', '15', '17.5', '20', '22.5'],fontsize=14)
    plt.yticks(fontsize=14)
                    
    num_cir = int(shape[0]*shape[1])
    for i in range(num_cir):
        x = spacing+spacing*(i % shape[0])
        y = spacing+spacing*(i % shape[1])
        ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
                    
    Z_n = calc_Z(X.flatten(),Y.flatten(),*params)
    Z_n = np.reshape(Z_n,X.shape)

    params = str(params)
    i0 = 15
    i1 = 16
    params = params[:i0]+r'$\bf{x}$'.format(x=params[i0:i1])+params[i1:] 

    ax.set_title(f'\u03B8 = {params}',size=16)
                    
    cont = ax.contourf(X,Y,Z_n,cmap=cm.coolwarm,vmin=0)
    
    plt.savefig('/home/thomas/Pictures/RAL_2/r.png',bbox_inches='tight')
    plt.show()