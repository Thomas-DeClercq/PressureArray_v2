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

from PressureReconstruction import calc_Z

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

#from generatePressureDistributionsInTime import animate


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
    plot_contour = False
    n = 50
    
    all_files = os.listdir("./Data5")
    #all_files.sort()
    split = 0.8
    split = round(0.8*len(all_files))
    training_files = all_files[:split]
    test_files = all_files[split:]
    
    error_array = np.zeros((len(all_files),2))
    #file = all_files[10]
    
    time_step = 1 #currently max of 1, can later be increased
    
    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    
    for file in training_files:
        df = pd.read_csv("./Data5/"+file)
        
        amount_of_timesteps = int(df["Timestep"].max()/time_step)
        
        for t in range(amount_of_timesteps):
            df_t = df.loc[df["Timestep"] >= t]
            df_t = df_t.loc[df_t["Timestep"] <= t+time_step]
            #important to check these df's when adding complexity
            
            try:
                x_train = np.vstack((x_train,df_t.values.T[2:10].T.flatten()))
                current_timestep = df_t["Timestep"].max()
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] - df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[0]
                y_train = np.vstack((y_train,angles))
            except:
                x_train = np.append(x_train,df_t.values.T[2:10].T.flatten())
                current_timestep = df_t["Timestep"].max()
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] - df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_train = np.append(y_train,angles)
 
    for file in test_files:
        df = pd.read_csv("./Data5/"+file)
        
        amount_of_timesteps = int(df["Timestep"].max()-time_step+1)
        
        for t in range(amount_of_timesteps):
            df_t = df.loc[df["Timestep"] >= t]
            df_t = df_t.loc[df_t["Timestep"] <= t+time_step]
            #important to check these df's when adding complexity
            
            try:
                x_test = np.vstack((x_test,df_t.values.T[2:10].T.flatten()))
                current_timestep = df_t["Timestep"].max()
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] - df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_test = np.vstack((y_test,angles))
            except:
                x_test = np.append(x_test,df_t.values.T[2:10].T.flatten())
                current_timestep = df_t["Timestep"].max()
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] - df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_test = np.append(y_test,angles)
    
    svr = SVR(kernel='linear')
    svr.fit(x_train,y_train)
    
    y_train_est = svr.predict(x_train)
    y_test_est = svr.predict(x_test)
    
    """
    error_array = np.append(y_train,y_test)
    error_array = np.vstack((error_array,np.append(y_train_est,y_test_est)))
    error_array = np.vstack((error_array,abs(error_array[1] - error_array[0])))
    """
    error_array = y_test.copy().flatten()
    error_array = np.vstack((error_array,y_test_est))
    error_array = np.vstack((error_array,abs(error_array[1] - error_array[0])))
    
    print(f"average: {np.average(error_array[2])}")
    print(f"median: {np.median(error_array[2])}")
    print(f"std: {np.std(error_array[2])}")
    """
    for idx,file in enumerate(all_files):
        df = pd.read_csv("./Data1/"+file)
        
        timesteps = int(df['Timestep'].max() + 1)
        it = int(df['iteration'].max() + 1)
        
        all_params = np.zeros((timesteps,it,8))
        all_params = df2numpy(all_params,df)
            
        real_diff = df['angle'][1]-df['angle'][0]
        
        error_array[idx][0] = real_diff
        error_array[idx][1] = 0
        
        if plot_contour:
            X = np.linspace(0, (shape[0]+1)*spacing, n)
            Y = np.linspace(0, (shape[1]+1)*spacing, n)
            X, Y = np.meshgrid(X, Y)
            
            t=0
            Z_n0 = np.zeros(X.shape)
            for itt in range(all_params.shape[1]):
                Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params[t][itt])
                Z_1 = np.reshape(Z_1,X.shape)
                Z_n0 = Z_n0 + Z_1
            
            t=1
            Z_n1 = np.zeros(X.shape)
            for itt in range(all_params.shape[1]):
                Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params[t][itt])
                Z_1 = np.reshape(Z_1,X.shape)
                Z_n1 = Z_n1 + Z_1
            fig1 = plt.figure(figsize=[(shape[0]+1)*2,(shape[1]+1)])
            ax2 = plt.subplot(121)
            ax2.set_xlim([0,spacing*(shape[0]+1)])
            ax2.set_ylim([0,spacing*(shape[1]+1)])
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')           
            ax2.contourf(X,Y,Z_n0,cmap=cm.coolwarm,vmin=0)
            
            ax3 = plt.subplot(122)
            ax3.set_xlim([0,spacing*(shape[0]+1)])
            ax3.set_ylim([0,spacing*(shape[1]+1)])
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')         
            ax3.contourf(X,Y,Z_n1,cmap=cm.coolwarm,vmin=0)
       """     