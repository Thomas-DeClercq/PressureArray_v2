#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:38:07 2022

@author: pc-robotiq
"""
import os
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm, animation
import matplotlib.patches as patches

from PressureReconstruction import calc_Z
#from generatePressureDistributionsInTime import animate

def animate(i,X,Y,all_params):
    global cont
    
    Z_n = np.zeros(X.shape)
    for it in range(all_params.shape[1]):
        Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params[i][it])
        Z_1 = np.reshape(Z_1,X.shape)
        Z_n = Z_n + Z_1
    
    for c in cont.collections:
        c.remove()
    cont = ax.contourf(X,Y,Z_n,cmap=cm.coolwarm,vmin=0)
    TXT_t.set_text(str("t = ")+str(i))
    return cont,

def df2numpy(params,df):
    for t in range(params.shape[0]):
        for iti in range(params.shape[1]):
            for i in range(params.shape[2]):
                row = t*params.shape[1] + iti
                params[t][iti][i] = df.iloc[row][i+2]
    
    return params

if __name__ == "__main__":
    
    shape = [5,9]
    spacing = 4.5
    n = 50
    
    all_params0 = [10,3,10,10,0,0,13.5,22.5]
    
    X = np.linspace(0, (shape[0]+1)*spacing, n)
    Y = np.linspace(0, (shape[1]+1)*spacing, n)
    X, Y = np.meshgrid(X, Y)
    
    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    
    num_cir = int(shape[0]*shape[1])
    for i in range(num_cir):
        x = spacing+spacing*(i % shape[0])
        y = spacing+spacing*(i % shape[1])
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
    
    Z_n = np.zeros(X.shape)
    Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params0)
    Z_1 = np.reshape(Z_1,X.shape)
    Z_n = Z_n + Z_1
    
    cont = ax.contourf(X,Y,Z_n,cmap=cm.coolwarm,vmin=0)
    fig.show()
        
        
        