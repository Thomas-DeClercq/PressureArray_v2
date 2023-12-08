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
from math import pi
import random

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

def doPCA(X,Y,Z_n):
    
    n = len(X)
    Z_prop = Z_n/(np.max(Z_n)*2)
    Z_bool = np.random.rand(n,) < Z_prop
    X_mask = X[Z_bool]
    Y_mask = Y[Z_bool]
    """
    Z_mask = np.ma.MaskedArray(Z_n, mask=Z_bool)
    X_mask = np.ma.MaskedArray(X, mask=Z_bool)
    Y_mask = np.ma.MaskedArray(Y, mask=Z_bool)
    """
    arr = np.array([X_mask,Y_mask])#,Z_mask])
    
    col = ['X', 'Y']#, 'pressure']
    df = pd.DataFrame(arr.T,columns=col)
    
    pca = PCA(n_components=2)
    pca.fit(df)
    comp = pca.components_
    #print(comp)
    
    #id_del = np.argmax(comp.T[2])
    #comp = np.delete(comp,id_del,axis=0)
    
    #calc lines starting in center distibution
    
    sum_x = 0
    sum_p = 0
    for i,x in enumerate(X):
        sum_x += x*Z_n[i]
        sum_p += Z_n[i]
    p0_x = sum_x/sum_p
    
    sum_y = 0
    sum_p = 0
    for i,y in enumerate(Y):
        sum_y += y*Z_n[i]
        sum_p += Z_n[i]
    p0_y = sum_y/sum_p
    
    
    p0 = (p0_x, p0_y)#, 0)
    p1i = p0 + 100*comp[0]
    p1e = p0 - 100*comp[0]
    
    p2i = p0 + 100*comp[1]
    p2e = p0 - 100*comp[1]
    """
    p3i = p0 + 100*comp[2]
    p3e = p0 - 100*comp[2]
    """
    l1 = tuple(zip(p1i,p1e))
    l2 = tuple(zip(p2i,p2e))
    """
    l3 = tuple(zip(p3i,p3e))
    """
    angle_1 = np.arctan2(comp[0][0],comp[0][1])/pi*180
    angle_2 = np.arctan2(comp[1][0],comp[1][1])/pi*180
    """
    angle_3 = np.arctan2(comp[2][0],comp[2][1])/pi*180
    
    return comp, l1, l2, l3, angle_1, angle_2, angle_3
    """
    return comp, l1, l2, angle_1, angle_2, X_mask, Y_mask
    
def findAngles(comp):
    #id_del = np.argmax(comp.T[2])
    comp_1 = comp
    #comp_1 = np.delete(comp,id_del,axis=0)
    
    angle_1 = np.arctan2(comp_1[0][0],comp_1[0][1])/pi*180
    angle_2 = np.arctan2(comp_1[1][0],comp_1[1][1])/pi*180
    angle_3 = 0 #np.arctan2(comp_1[2][0],comp_1[2][1])/pi*180
    return angle_1, angle_2, angle_3

if __name__ == "__main__":
    
    shape = [5,9]
    spacing = 4.5
    plot_scatter = False
    plot_contour = False
    n = 100
    timesteps = 1
    
    data = 'Data8'
    
    all_files = os.listdir("./"+data)
    #random.shuffle(all_files)
    all_files = all_files[:500]
    #all_files.sort()

    error_array = np.zeros((len(all_files)*timesteps,7))
    for idx,file in enumerate(all_files):
    #error_array = np.zeros((1*timesteps,7))
    #if True:    
        #file = all_files[5]
        #idx = 0
        df = pd.read_csv("./"+data+"/"+file)
        
        timesteps = int(df['Timestep'].max() + 1)
        it = int(df['iteration'].max() + 1)
        
        all_params = np.zeros((timesteps,it,8))
        all_params = df2numpy(all_params,df)
        
        X = np.linspace(0, (shape[0]+1)*spacing, n)
        Y = np.linspace(0, (shape[1]+1)*spacing, n)
        X, Y = np.meshgrid(X, Y)
        X = X.flatten()
        Y = Y.flatten()
        
        for ti in range(int(df["Timestep"].max())):
        #if True:
            #ti=0
            t=ti
            Z_n0 = np.zeros(X.shape)
            for itt in range(all_params.shape[1]):
                Z_1 = calc_Z(X,Y,*all_params[t][itt])
                Z_n0 = Z_n0 + Z_1
            
            #comp0, l01, l02, l03, angle_p01, angle_p02, angle_p03 = doPCA(X,Y,Z_n0)            
            #print(angle_p01,angle_p02)
            comp0, l01, l02, angle_p01, angle_p02, X1, Y1 = doPCA(X,Y,Z_n0)
            
            
            t=ti+1
            Z_n1 = np.zeros(X.shape)
            for itt in range(all_params.shape[1]):
                Z_1 = calc_Z(X,Y,*all_params[t][itt])
                Z_n1 = Z_n1 + Z_1
            
            #comp1, l11, l12, l13, angle_p11, angle_p12, angle_p13 = doPCA(X,Y,Z_n1)            
            #print(angle_p11,angle_p12)
            comp1, l11, l12, angle_p11, angle_p12, X2, Y2 = doPCA(X,Y,Z_n1)
            
            #print('-----------------')
            diff_p1 = angle_p11 - angle_p01
            diff_p2 = angle_p12 - angle_p02
            #diff_p3 = angle_p13 - angle_p03
            
            while diff_p1 > 90:
                diff_p1 -= 180
            while diff_p1 < -90:
                diff_p1 += 180
            
            while diff_p2 > 90:
                diff_p2 -= 180
            while diff_p2 < -90:
                diff_p2 += 180
            """
            while diff_p3 > 90:
                diff_p3 -= 180
            while diff_p3 < -90:
                diff_p3 += 180
            """
            real_diff = df.loc[df["Timestep"]==ti+1]['angle'].values[0] - df.loc[df["Timestep"]==ti]['angle'].values[0]#df['angle'][ti+1]-df['angle'][ti]
            
            #print(diff_p1, diff_p2)#, diff_p3)
            #print(f"real diff: {real_diff}")
            
            timesteps -= 1
            error_array[idx*timesteps+ti][0] = real_diff
            error_array[idx*timesteps+ti][1] = diff_p1
            error_array[idx*timesteps+ti][2] = diff_p2
            #error_array[idx*timesteps+ti][3] = diff_p3
            error_array[idx*timesteps+ti][4] = -real_diff-diff_p1
            error_array[idx*timesteps+ti][5] = -real_diff-diff_p2
            #error_array[idx*timesteps+ti][6] = -real_diff-diff_p3
        
        if plot_scatter:
            fig = plt.figure(figsize=[(shape[0]+1)*2,(shape[1]+1)])
            ax = plt.subplot(121)
            ax.set_xlim([0,spacing*(shape[0]+1)])
            ax.set_ylim([0,spacing*(shape[1]+1)])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            ax.scatter(X1,Y1,s=1)
            ax.plot(l01[0], l01[1], color='g')
            ax.plot(l02[0], l02[1], color='k')
            #ax.plot(l03[0], l03[1], l03[2])
            
            ax1 = plt.subplot(122)
            ax1.set_xlim([0,spacing*(shape[0]+1)])
            ax1.set_ylim([0,spacing*(shape[1]+1)])
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            
            ax1.scatter(X2,Y2,s=1)
            ax1.plot(l11[0], l11[1],color='g')
            ax1.plot(l12[0], l12[1],color='k')
            #ax1.plot(l13[0], l13[1], l13[2])
            
            
        if plot_contour:
            X = np.reshape(X,(n,n))
            Y = np.reshape(Y,(n,n))
            Z_n0 = np.reshape(Z_n0,(n,n))
            Z_n1 = np.reshape(Z_n1,(n,n))
            
            fig1 = plt.figure(figsize=[(shape[0]+1)*2,(shape[1]+1)])
            ax2 = plt.subplot(121)
            ax2.set_xlim([0,spacing*(shape[0]+1)])
            ax2.set_ylim([0,spacing*(shape[1]+1)])
            #ax2.set_zlim([0,np.max(Z_n0)])
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            #ax2.set_zlabel('p')
            
            ax2.contourf(X,Y,Z_n0,cmap=cm.coolwarm,vmin=0)
            ax2.plot(l01[0], l01[1],color='g')
            ax2.plot(l02[0], l02[1],color='k')
            #ax2.plot(l03[0], l03[1],color='y')
            
            ax3 = plt.subplot(122)
            ax3.set_xlim([0,spacing*(shape[0]+1)])
            ax3.set_ylim([0,spacing*(shape[1]+1)])
            #ax3.set_zlim([0,np.max(Z_n1)])
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            #ax3.set_zlabel('p')
            
            ax3.contourf(X,Y,Z_n1,cmap=cm.coolwarm,vmin=0)
            ax3.plot(l11[0], l11[1],color='g')
            ax3.plot(l12[0], l12[1],color='k')
            #ax3.plot(l13[0], l13[1],color='y')
      
    print("##############################################")
    print(f"average error p1: {np.average(abs(error_array.T[4]))}")
    print(f"average error p2: {np.average(abs(error_array.T[5]))}")
    #print(f"average error p3: {np.average(abs(error_array.T[6]))}")