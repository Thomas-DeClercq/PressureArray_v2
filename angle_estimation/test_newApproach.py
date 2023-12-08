#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:35:22 2023

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
import time

import scipy
from scipy.interpolate import RBFInterpolator, Rbf

from PressureReconstruction import calc_Z, Optimization

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pickle
#pickle.dump(svr,open('./numpy_arrays/svr_STEP4_40_180_02i.sav','wb'))

#from generatePressureDistributionsInTime import animate


def df2numpy(params,df):
    for t in range(params.shape[0]):
        for iti in range(params.shape[1]):
            for i in range(params.shape[2]):
                row = t*params.shape[1] + iti
                params[t][iti][i] = df.iloc[row][i+2]
    
    return params

def getValue(X,Y,Z,xi,yi):
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    point = list(zip(xi,yi))
    #point = [(xi,yi)]
    zi = scipy.interpolate.griddata((X,Y),Z,point)
    zi = np.nan_to_num(zi)
    return zi

def pressureReconstruction_df(df,shape,spacing,n=50,m=10,p_m=0.75):
    X = np.linspace(0, (shape[0]+1)*spacing, n)
    Y = np.linspace(0, (shape[1]+1)*spacing, n)
    X, Y = np.meshgrid(X, Y)
    
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
    
    timesteps = int(df['Timestep'].max() + 1)
    it_in = int(df['iteration'].max() + 1)
    it_out = 5
    
    list_x0 = np.zeros((it_out,8))
    df_new = np.zeros((timesteps*it_out,11))
    
    for t in range(timesteps):
        Z_n = np.zeros(X.shape)
        for i in range(it_in):
            params_i = df.loc[df['Timestep']==t].loc[df['iteration']==i].values[0][2:10]
            Z_1 = calc_Z(X.flatten(),Y.flatten(),*params_i)
            Z_1 = np.reshape(Z_1,X.shape)
            Z_n = Z_n + Z_1
        
        Z_i = getValue(X, Y, Z_n, xi, yi)
        
        rbfi = Rbf(xi,yi,Z_i,function='gaussian') #always +-2.5 ms
        Z_rbf = rbfi(X_rbf,Y_rbf)
        array_rbf = np.array([X_rbf.flatten(),Y_rbf.flatten(),Z_rbf.flatten()]).T
    
        idxs = []
        for idx,el in enumerate(array_rbf):
            if el[0] < spacing or el[0] > shape[0]*spacing:
                idxs.append(idx)
            elif el[1] < spacing or el[1] > shape[1]*spacing:
                idxs.append(idx)
        array_rbf = np.delete(array_rbf,idxs,axis=0).T
        array, list_E = Optimization(xi,yi,Z_i.copy(),shape,spacing, array_rbf, list_x0, n=round(p_m*array_rbf.shape[1]),it_max=it_out,t_max=50)
        list_x0 = np.zeros((it_out,8))
        for idx,E in enumerate(list_E):
            list_x0[idx] = E.x
            
        for idx1 in range(it_out):
            df_new[t*it_out+idx1,0] = t
            df_new[t*it_out+idx1,1] = idx1
            df_new[t*it_out+idx1,10] = df.loc[df['Timestep']==t].values[-1][-1]
            for idx2 in range(8):
                df_new[t*it_out+idx1,2+idx2] = list_x0[idx1,idx2]
                
    columns = ['Timestep','iteration','p0','std','lx','ly','r_curve','theta','x0','y0','angle']
    df_new = pd.DataFrame(df_new,columns=columns)        
    return df_new

if __name__ == "__main__":
    
    t0 = time.time()
    shape = [5,9]
    spacing = 4.5
    plot = True
    n = 50
    data = "STEP4_40_180"
    
    all_files = os.listdir("./"+data)
    random.shuffle(all_files)
    test_files = all_files[:10]
    
    error_array = np.zeros((len(test_files),2))
    #file = test_files[10]
    
    time_steps = 1 #currently max of 1, can later be increased --> doens't improve accuracy, training time x2
    
    x_test = np.array([])
    y_test = np.array([])
    y_est  = np.array([])
    
    #StdS_X = StandardScaler()
    #StdS_Y = StandardScaler()
    
    svr = pickle.load(open('./numpy_arrays/svr_STEP4_40_180_newAproach.sav','rb'))
    
    print("converting test data")
    for file in test_files:
        df = pd.read_csv("./"+data+"/"+file)
        
        df = pressureReconstruction_df(df,shape,spacing)
        
        amount_of_timesteps = int(df["Timestep"].max()-time_steps+1)
        
        for t in range(amount_of_timesteps):
            df_t0 = df.loc[df["Timestep"] == 0]
            df_ti = df.loc[df["Timestep"] >= t]
            df_ti = df_ti.loc[df_ti["Timestep"] <= t+time_steps-1]
            #df_ti = df.loc[df["Timestep"] == t]
            df_t = pd.concat([df_t0,df_ti])
            #df_t = df.loc[df["Timestep"] >= t]
            #df_t = df_t.loc[df_t["Timestep"] <= t+time_steps]
            #important to check these df's when adding complexity
            
            try:
                current_timestep = df_t["Timestep"].max()
                if t == 0:
                    prev_angle = 0
                else:
                    prev_angle = y_test[-1]
                x_test_new = np.append(df_t.values.T[2:10].T.flatten(),prev_angle)
                x_test = np.vstack((x_test,x_test_new))
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_test = np.vstack((y_test,angles))
                y_est_new = svr.predict(x_test_new.reshape(1,-1))
                y_est = np.vstack((y_est,y_est_new))
            except: #only for the first time ever
                current_timestep = df_t["Timestep"].max()
                prev_angle = 0
                x_test_new = np.append(df_t.values.T[2:10].T.flatten(),prev_angle)
                x_test = np.append(x_test,x_test_new)
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_test = np.append(y_test,angles)
                y_est_new = svr.predict(x_test_new.reshape(1,-1))
                y_est = np.append(y_est,y_est_new)
    
    #x_train = StdS_X.fit_transform(x_train)
    #y_train = StdS_Y.fit_transform(y_train)
    
    """
    for idx,el in enumerate(y_test_est):
        if el > 90:
            y_test_est[idx] = el- 180
        elif el < -90:
            y_test_est[idx] = el + 180
    """
    print("error calculation")
    error_array_rel = y_test.copy().flatten()
    error_array_rel = np.vstack((error_array_rel,y_est.flatten()))
    error_array_rel = np.vstack((error_array_rel,abs(error_array_rel[1] - error_array_rel[0])))
    
    """
    o = len(test_files)
    m = len(y_test)/o
    if m%1 != 0:
        print("error!!!")
    else:
        m = int(m)
        
    
    error_array_abs = np.zeros(error_array_rel.shape)
    for i in range(o):
        for j in range(m):
            if j == 0:
                error_array_abs[0][i*m+j] = y_test[i*m+j]
                error_array_abs[1][i*m+j] = y_test_est[i*m+j]
            else:
                error_array_abs[0][i*m+j] = error_array_abs[0][i*m+j-1] + y_test[i*m+j]
                error_array_abs[1][i*m+j] = error_array_abs[1][i*m+j-1] + y_test_est[i*m+j]
                
    error_array_abs[2] = abs(error_array_abs[1] - error_array_abs[0])
    """
    print("rel")
    print(f"average: {np.average(error_array_rel[2])}")
    print(f"median: {np.median(error_array_rel[2])}")
    print(f"std: {np.std(error_array_rel[2])}")
    """
    print('-----------------------------------------')
    print("abs")
    print(f"average: {np.average(error_array_abs[2])}")
    print(f"median: {np.median(error_array_abs[2])}")
    print(f"std: {np.std(error_array_abs[2])}")
    """
    
    if plot:
        timesteps = 39
        for i in range(20):
            fig1 = plt.figure()
            ax = plt.subplot(111)
            ax.plot(error_array_rel[0][i*timesteps:(i+1)*timesteps-1],label='real angle')
            ax.plot(error_array_rel[1][i*timesteps:(i+1)*timesteps-1],label='estimated angle')
            ax.legend()