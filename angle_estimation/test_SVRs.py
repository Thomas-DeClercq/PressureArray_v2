#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:35:22 2023

@author: pc-robotiq
"""
import os
import pandas as pd
import numpy as np

#from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib import cm, animation
import matplotlib.patches as patches
from math import pi
import random
import time

import scipy
from scipy.interpolate import RBFInterpolator, Rbf

import sys
sys.path.insert(1,'/home/thomas/pythonScripts/PressureArray_v2/PR_cython')
from PressureReconstruction_update210623 import calc_Z, Optimization

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
    it_out = 1
    
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
    shape = [4,8]
    spacing = 4.5
    plot = True
    n = 50
    all_data = ["Data_realistic_test"]
    all_svrs = ["Data_realistic_1k"]
    Scalers = False

    for data in all_data:
        for svr_name in all_svrs:
            print(f"SVR: {svr_name}, data: {data}")
            """
            if svr_name == all_svrs[-1] and data == all_data[0]:
                continue
            """
            all_files = os.listdir("./Data_RT/TrainingData/"+data)
            #random.shuffle(all_files)
            random.shuffle(all_files)
            test_files = all_files[:]

            
            error_array = np.zeros((len(test_files),2))
            #file = test_files[10]
            """
            if svr_name == all_svrs[-1]:
                time_steps = 2
            else:
                time_steps = 1 #currently max of 1, can later be increased --> doens't improve accuracy, training time x2
            """
            time_steps = 1
            x_test = np.array([])
            y_test = np.array([])
            y_est  = np.array([])
            
            #StdS_X = StandardScaler()
            #StdS_Y = StandardScaler()
            
            #svr = pickle.load(open(f'./numpy_arrays/svr_{svr_name}.sav','rb'))
            #svr = pickle.load(open(f'./Data_RT/SVR/{svr_name}.sav','rb'))
            if Scalers:
                StdS_X = pickle.load(open(f'./Data_RT/SVR/{svr_name}_StdS_X.sav','rb'))
                #StdS_Y = pickle.load(open(f'./Data_RT/SVR/{svr_name}_StdS_Y.sav','rb'))
                svr = pickle.load(open(f'./Data_RT/SVR/{svr_name}_StdS.sav','rb'))
            else:
                svr = pickle.load(open(f'./Data_RT/SVR/{svr_name}.sav','rb'))

            progress_counter = 0.1
            print("converting test data")
            for idx,file in enumerate(test_files):
                if (idx/len(test_files)) >= progress_counter:
                    print(f'{idx/len(test_files)*100}% done')
                    progress_counter = progress_counter + 0.1
                #df = pd.read_csv("./"+data+"/"+file)
                df = pd.read_csv(f"./Data_RT/TrainingData/{data}/{file}")
                
                df = pressureReconstruction_df(df,shape,spacing)
                
                amount_of_timesteps = int(df["Timestep"].max()-time_steps+1)
                
                for t in range(0,amount_of_timesteps+1):
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
                        x_test_new = df_t.values.T[2:10].T.flatten()
                        x_test = np.vstack((x_test,x_test_new))
                        angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                        #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                        y_test = np.vstack((y_test,angles))
                        if Scalers:
                            x_test_new_transform = StdS_X.transform(x_test_new.reshape(1,-1))
                            y_est_new = svr.predict(x_test_new_transform) #.reshape(1,-1)
                            #y_est_new = StdS_Y.inverse_transform(y_est_new.reshape(1,-1)) 
                        else:
                            y_est_new = svr.predict(x_test_new.reshape(1,-1))
                        y_est = np.vstack((y_est,y_est_new))
                    except: #only for the first time ever
                        current_timestep = df_t["Timestep"].max()
                        x_test_new = df_t.values.T[2:10].T.flatten()
                        x_test = np.append(x_test,x_test_new)
                        angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                        #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                        y_test = np.append(y_test,angles)
                        if Scalers:
                            x_test_new_transform = StdS_X.transform(x_test_new.reshape(1,-1))
                            y_est_new = svr.predict(x_test_new_transform) #.reshape(1,-1)
                            #y_est_new = StdS_Y.inverse_transform(y_est_new.reshape(1,-1)) 
                        else:
                            y_est_new = svr.predict(x_test_new.reshape(1,-1))
                        y_est = np.append(y_est,y_est_new)
            
            #x_train = StdS_X.fit_transform(x_train)
            #y_train = StdS_Y.fit_transform(y_train)
            #np.savetxt('./Data_RT/TrainingData/x_test.txt', x_test, fmt='%d')
            #np.savetxt('./Data_RT/TrainingData/y_test.txt', y_test, fmt='%d')
            
            for idx,el in enumerate(y_est):
                while el > 180:
                    y_est[idx] = el- 180
                while el < -90:
                    y_est[idx] = el + 180
            
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
            #print(f"SVR: {svr_name}, data: {data}")
            print(f"average: {np.average(error_array_rel[2])}")
            print(f"median: {np.median(error_array_rel[2])}")
            print(f"std: {np.std(error_array_rel[2])}")
            np.save(f'./numpy_arrays/all_error_array_svr_{svr_name}_data_{data}.npy',error_array_rel)
            """
            print('-----------------------------------------')
            print("abs")
            print(f"average: {np.average(error_array_abs[2])}")
            print(f"median: {np.median(error_array_abs[2])}")
            print(f"std: {np.std(error_array_abs[2])}")
            """
            
            if plot:
                #timesteps = int(data.split('_')[1])
                timesteps = 10
                for i in range(4):
                    fig1 = plt.figure()
                    ax = plt.subplot(111)
                    ax.plot(error_array_rel[0][i*timesteps:(i+1)*timesteps-1],label='real angle')
                    ax.plot(error_array_rel[1][i*timesteps:(i+1)*timesteps-1],label='estimated angle')
                    ax.legend()
                    plt.show(block=False)

                fig2 = plt.figure()
                ax2 = plt.subplot(211)
                ax2.plot(error_array_rel[0],error_array_rel[1],'.')
                ax2.set_xlabel('real angle')
                ax2.set_ylabel('estimated angle')
                ax2.plot([0,90],[0,90],color='k')
                ax3 = plt.subplot(212)
                ax3.boxplot(error_array_rel[2], showfliers=False)
                plt.show(block=False)

                input()
