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

import scipy
from scipy.interpolate import RBFInterpolator, Rbf

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

def doPCA(X,Y,Z_n,p=0.5):
    
    Z_bool = Z_n > p*np.max(Z_n)
    Z_mask = Z_n[Z_bool]
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
    return comp, l1, l2, angle_1, angle_2
    
def findAngles(comp):
    #id_del = np.argmax(comp.T[2])
    comp_1 = comp
    #comp_1 = np.delete(comp,id_del,axis=0)
    
    angle_1 = np.arctan2(comp_1[0][0],comp_1[0][1])/pi*180
    angle_2 = np.arctan2(comp_1[1][0],comp_1[1][1])/pi*180
    angle_3 = 0 #np.arctan2(comp_1[2][0],comp_1[2][1])/pi*180
    return angle_1, angle_2, angle_3

def getValue(X,Y,Z,xi,yi):
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    point = list(zip(xi,yi))
    #point = [(xi,yi)]
    zi = scipy.interpolate.griddata((X,Y),Z,point)
    zi = np.nan_to_num(zi)
    return zi

if __name__ == "__main__":
    
    shape = [5,9]
    spacing = 4.5
    plot_scatter = False
    plot_contour = False
    n = 50
    m = 50
    timesteps = 40
    p=0.5
    
    data = 'STEP4_40_180'
    
    all_files = os.listdir("./"+data)
    random.shuffle(all_files)
    #all_files = all_files[:200]
    #all_files.sort()

    error_array = np.zeros((len(all_files)*(timesteps-1),7))
    row_idx = 0
    plot_counter = 0
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
        
        X_rbf = np.linspace(0, (shape[0]+1)*spacing, m)
        Y_rbf = np.linspace(0, (shape[1]+1)*spacing, m)
        X_rbf, Y_rbf = np.meshgrid(X_rbf, Y_rbf)
        
        xi = []
        yi = []
        for x_i in range(shape[0]):
            for y_i in range(shape[1]):
                xi.append(spacing+x_i*spacing)
                yi.append(spacing+y_i*spacing)
                
        xi = np.array(xi)
        yi = np.array(yi)
        
        for ti in range(int(df["Timestep"].max())):
        #if True:
            #ti=0
            t=0
            Z_n0 = np.zeros(X.shape)
            for itt in range(all_params.shape[1]):
                Z_1 = calc_Z(X,Y,*all_params[t][itt])
                Z_n0 = Z_n0 + Z_1
            
            Z_i0 = getValue(X, Y, Z_n0, xi, yi)
            
            rbfi0 = Rbf(xi,yi,Z_i0,function='gaussian') #always +-2.5 ms
            Z_rbf0 = rbfi0(X_rbf,Y_rbf)
            
            #comp0, l01, l02, l03, angle_p01, angle_p02, angle_p03 = doPCA(X,Y,Z_n0)            
            #print(angle_p01,angle_p02)
            comp0, l01, l02, angle_p01, angle_p02 = doPCA(X_rbf.flatten(),Y_rbf.flatten(),Z_rbf0.flatten(),p=p)
            
            
            t=ti+1
            Z_n1 = np.zeros(X.shape)
            for itt in range(all_params.shape[1]):
                Z_1 = calc_Z(X,Y,*all_params[t][itt])
                Z_n1 = Z_n1 + Z_1
                
                
            Z_i1 = getValue(X, Y, Z_n1, xi, yi)
            
            rbfi1 = Rbf(xi,yi,Z_i1,function='gaussian') #always +-2.5 ms
            Z_rbf1 = rbfi1(X_rbf,Y_rbf)
            
            #comp1, l11, l12, l13, angle_p11, angle_p12, angle_p13 = doPCA(X,Y,Z_n1)            
            #print(angle_p11,angle_p12)
            comp1, l11, l12, angle_p11, angle_p12 = doPCA(X_rbf.flatten(),Y_rbf.flatten(),Z_rbf1.flatten(),p=p)
            
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
            real_diff = df.loc[df["Timestep"]==ti]['angle'].values[0] #- df.loc[df["Timestep"]==ti]['angle'].values[0]#df['angle'][ti+1]-df['angle'][ti]
            
            #print(diff_p1, diff_p2)#, diff_p3)
            #print(f"real diff: {real_diff}")
            
            #timesteps -= 1
            error_array[row_idx][0] = real_diff
            error_array[row_idx][1] = diff_p1
            error_array[row_idx][2] = diff_p2
            #error_array[row_idx][3] = diff_p3
            error_array[row_idx][4] = -real_diff-diff_p1
            error_array[row_idx][5] = -real_diff-diff_p2
            #error_array[row_idx][6] = -real_diff-diff_p3
            row_idx += 1
        
        
        if plot_scatter and (plot_counter < 20):
            fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)*3])
            ax = plt.subplot(121,projection='3d')
            ax.set_xlim([0,spacing*(shape[0]+1)])
            ax.set_ylim([0,spacing*(shape[1]+1)])
            ax.set_zlim([0,np.max(Z_n0)])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('p')
            
            ax.scatter3D(X,Y,Z_n0)
            ax.plot(l01[0], l01[1], (0,0))
            ax.plot(l02[0], l02[1], (0,0))
            #ax.plot(l03[0], l03[1], l03[2])
            
            ax1 = plt.subplot(122,projection='3d')
            ax1.set_xlim([0,spacing*(shape[0]+1)])
            ax1.set_ylim([0,spacing*(shape[1]+1)])
            ax1.set_zlim([0,np.max(Z_n1)])
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('p')
            
            ax1.scatter3D(X,Y,Z_n1)
            ax1.plot(l11[0], l11[1], (0,0))
            ax1.plot(l12[0], l12[1], (0,0))
            #ax1.plot(l13[0], l13[1], l13[2])
            
            plot_counter += 1
        
            
            
        if plot_contour and (plot_counter < 20):
            X = np.reshape(X,(n,n))
            Y = np.reshape(Y,(n,n))
            Z_n0 = np.reshape(Z_n0,(n,n))
            Z_n1 = np.reshape(Z_n1,(n,n))
            num_cir = int(shape[0]*shape[1])
            
            fig1 = plt.figure(figsize=[(shape[0]+1)*2,(shape[1]+1)*2])
            ax1 = plt.subplot(221)
            ax1.set_xlim([0,spacing*(shape[0]+1)])
            ax1.set_ylim([0,spacing*(shape[1]+1)])
            #ax1.set_zlim([0,np.max(Z_n0)])
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            #ax1.set_zlabel('p')
            ax1.set_aspect(1)
            ax1.set_title('Input t=0')
            
            ax1.contourf(X,Y,Z_n0,cmap=cm.coolwarm,vmin=0)
            ax1.plot(l01[0], l01[1],color='g')
            ax1.plot(l02[0], l02[1],color='k')
            #ax1.plot(l03[0], l03[1],color='y')
            
            for i in range(num_cir):
                x = spacing+spacing*(i % shape[0])
                y = spacing+spacing*(i % shape[1])
                ax1.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
            
            ax2 = plt.subplot(222)
            ax2.set_xlim([0,spacing*(shape[0]+1)])
            ax2.set_ylim([0,spacing*(shape[1]+1)])
            #ax2.set_zlim([0,np.max(Z_n1)])
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            #ax2.set_zlabel('p')
            ax2.set_aspect(1)
            ax2.set_title('Input t=1')
            
            ax2.contourf(X,Y,Z_n1,cmap=cm.coolwarm,vmin=0)
            ax2.plot(l11[0], l11[1],color='g')
            ax2.plot(l12[0], l12[1],color='k')
            #ax2.plot(l13[0], l13[1],color='y')
            
            for i in range(num_cir):
                x = spacing+spacing*(i % shape[0])
                y = spacing+spacing*(i % shape[1])
                ax2.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
            
            ax3 = plt.subplot(223)
            ax3.set_xlim([0,spacing*(shape[0]+1)])
            ax3.set_ylim([0,spacing*(shape[1]+1)])
            #ax3.set_zlim([0,np.max(Z_n0)])
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            #ax3.set_zlabel('p')
            ax3.set_aspect(1)
            ax3.set_title('RBF t=0')
            
            ax3.contourf(X_rbf,Y_rbf,Z_rbf0,cmap=cm.coolwarm,vmin=0)
            ax3.plot(l01[0], l01[1],color='g')
            ax3.plot(l02[0], l02[1],color='k')
            #ax3.plot(l03[0], l03[1],color='y')
            
            for i in range(num_cir):
                x = spacing+spacing*(i % shape[0])
                y = spacing+spacing*(i % shape[1])
                ax3.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
            
            ax4 = plt.subplot(224)
            ax4.set_xlim([0,spacing*(shape[0]+1)])
            ax4.set_ylim([0,spacing*(shape[1]+1)])
            #ax4.set_zlim([0,np.max(Z_n1)])
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            #ax4.set_zlabel('p')
            
            ax4.contourf(X_rbf,Y_rbf,Z_rbf1,cmap=cm.coolwarm,vmin=0)
            ax4.plot(l11[0], l11[1],color='g')
            ax4.plot(l12[0], l12[1],color='k')
            #ax4.plot(l13[0], l13[1],color='y')
            ax4.set_aspect(1)
            ax4.set_title('RBF t=1')
            
            for i in range(num_cir):
                x = spacing+spacing*(i % shape[0])
                y = spacing+spacing*(i % shape[1])
                ax4.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
            
            plot_counter += 1
      
    print("##############################################")
    print(f"average error p1: {np.average(abs(error_array.T[4]))}")
    print(f"median error p1: {np.median(abs(error_array.T[4]))}")
    #print(f"average error p3: {np.average(abs(error_array.T[6]))}")