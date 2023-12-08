#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:38:07 2022

@author: pc-robotiq
"""
import os
import pandas as pd
import numpy as np

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib import cm, animation
import matplotlib.patches as patches

from PressureReconstruction import calc_Z

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    plot_scatter = True
    n = 50
    
    all_files = os.listdir("./Data1")
    all_files.sort()
    file = all_files[0]
    
    df = pd.read_csv("./Data1/"+file)
    
    timesteps = int(df['Timestep'].max() + 1)
    it = int(df['iteration'].max() + 1)
    
    all_params = np.zeros((timesteps,it,8))
    all_params = df2numpy(all_params,df)
    
    X = np.linspace(0, (shape[0]+1)*spacing, n)
    Y = np.linspace(0, (shape[1]+1)*spacing, n)
    X, Y = np.meshgrid(X, Y)
    X = X.flatten()
    Y = Y.flatten()

    t=1
    Z_n = np.zeros(X.shape)
    for itt in range(all_params.shape[1]):
        Z_1 = calc_Z(X,Y,*all_params[t][itt])
        Z_n = Z_n + Z_1
    
    Z_bool = Z_n > 0.01*np.max(Z_n)
    Z_mask = Z_n[Z_bool]
    X_mask = X[Z_bool]
    Y_mask = Y[Z_bool]
    """
    Z_mask = np.ma.MaskedArray(Z_n, mask=Z_bool)
    X_mask = np.ma.MaskedArray(X, mask=Z_bool)
    Y_mask = np.ma.MaskedArray(Y, mask=Z_bool)
    """
    arr = np.array([X_mask,Y_mask,Z_mask])
    col = ['X', 'Y', 'pressure']
    df = pd.DataFrame(arr.T,columns=col)
    
    pca = PCA(n_components=3)
    pca.fit(df)
    comp = pca.components_
    print(comp)
    
    #calc lines starting in (13.5, 22.5,0)
    p0 = (13.5, 22.5, 0)
    p1i = p0 + 100*comp[0]
    p1e = p0 - 100*comp[0]
    
    p2i = p0 + 100*comp[1]
    p2e = p0 - 100*comp[1]
    
    p3i = p0 + 100*comp[2]
    p3e = p0 - 100*comp[2]
    
    l1 = tuple(zip(p1i,p1e))
    l2 = tuple(zip(p2i,p2e))
    l3 = tuple(zip(p3i,p3e))
    
    if plot_scatter:
        fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
        ax = plt.axes(projection='3d')
        ax.set_xlim([0,spacing*(shape[0]+1)])
        ax.set_ylim([0,spacing*(shape[1]+1)])
        ax.set_zlim([0,np.max(Z_mask)])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('p')
        
        ax.scatter3D(X_mask,Y_mask,Z_mask)
        ax.plot(l1[0], l1[1], l1[2])
        ax.plot(l2[0], l2[1], l2[2])
        ax.plot(l3[0], l3[1], l3[2])
        