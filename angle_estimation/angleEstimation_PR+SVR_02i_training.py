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

def pressureReconstruction_df(df,shape,spacing,n=50,m=20,p_m=0.75):
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
    plot_contour = False
    n = 50
    data_name = 'Data_realistic'
    Scaler = False
    save = True
    
    all_files = os.listdir(f"./Data_RT/TrainingData/{data_name}")
    random.shuffle(all_files)
    training_files = all_files[:1000]
    
    #error_array = np.zeros((len(all_files),2))
    #file = all_files[10]
    
    time_steps = 1 #currently max of 1, can later be increased --> doens't improve accuracy, training time x2
    
    x_train = np.array([])
    y_train = np.array([])

    if Scaler:
        StdS_X = StandardScaler()
        #StdS_Y = MinMaxScaler(clip=True)
    
    print("converting training data")
    progress_counter = 0.1
    for idx,file in enumerate(training_files):
        if (idx/len(training_files)) >+ progress_counter:
            print(f'{idx/len(training_files)*100}% done')
            progress_counter = progress_counter + 0.1
        df_input = pd.read_csv(f"./Data_RT/TrainingData/{data_name}/{file}")
        
        df = pressureReconstruction_df(df_input,shape,spacing)
        
        amount_of_timesteps = int(df["Timestep"].max()-time_steps+1)
        
        for t in range(0,amount_of_timesteps+1):
            df_t0 = df.loc[df["Timestep"] == 0]
            #df_ti = df.loc[df["Timestep"] == t]
            df_ti = df.loc[df["Timestep"] >= t]
            df_ti = df_ti.loc[df_ti["Timestep"] <= t+time_steps-1]
            df_t = pd.concat([df_t0,df_ti])
            #df_t = df_t.loc[df_t["Timestep"] <= t+time_steps]
            #important to check these df's when adding complexity
            
            try:
                current_timestep = df_t["Timestep"].max()
                x_train_new = df_t.values.T[2:10].T.flatten()
                x_train = np.vstack((x_train,x_train_new))
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[0]
                y_train = np.vstack((y_train,angles))
            except:
                current_timestep = df_t["Timestep"].max()
                x_train_new = df_t.values.T[2:10].T.flatten()
                x_train = np.append(x_train,x_train_new)
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_train = np.append(y_train,angles)

        if (plot_contour) and (idx < 20):
            n = 20
            X = np.linspace(0, (shape[0]+1)*spacing, n)
            Y = np.linspace(0, (shape[1]+1)*spacing, n)
            X, Y = np.meshgrid(X, Y)
            
            t=0
            Z_n0 = np.zeros(X.shape)
            for itt in range(int(df_input['iteration'].max()+1)):
                params_i = df_input.loc[df_input['Timestep']==t].loc[df_input['iteration']==itt].values[0][2:10]
                if np.sum(abs(params_i)) == 0:
                    break
                Z_1 = calc_Z(X.flatten(),Y.flatten(),*params_i)
                Z_1 = np.reshape(Z_1,X.shape)
                Z_n0 = Z_n0 + Z_1
                
            Z_n0_PR = np.zeros(X.shape)
            for itt in range(int(df['iteration'].max())+1):
                params_i = df.loc[df['Timestep']==t].loc[df['iteration']==itt].values[0][2:10]
                if np.sum(abs(params_i)) == 0:
                    break
                Z_1_PR = calc_Z(X.flatten(),Y.flatten(),*params_i)
                Z_1_PR= np.reshape(Z_1_PR,X.shape)
                Z_n0_PR = Z_n0_PR + Z_1_PR
            
            t=amount_of_timesteps
            Z_n1 = np.zeros(X.shape)
            for itt in range(int(df_input['iteration'].max()+1)):
                params_i = df_input.loc[df_input['Timestep']==t].loc[df_input['iteration']==itt].values[0][2:10]
                if np.sum(abs(params_i)) == 0:
                    break
                Z_1 = calc_Z(X.flatten(),Y.flatten(),*params_i)
                Z_1 = np.reshape(Z_1,X.shape)
                Z_n1 = Z_n1 + Z_1
                
            Z_n1_PR = np.zeros(X.shape)
            for itt in range(int(df['iteration'].max()+1)):
                params_i = df.loc[df['Timestep']==t].loc[df['iteration']==itt].values[0][2:10]
                if np.sum(abs(params_i)) == 0:
                    break
                Z_1_PR = calc_Z(X.flatten(),Y.flatten(),*params_i)
                Z_1_PR= np.reshape(Z_1_PR,X.shape)
                Z_n1_PR = Z_n1_PR + Z_1_PR
                
            fig1 = plt.figure(figsize=[(shape[0]+1)*2,(shape[1]+1)*2])
            ax2 = plt.subplot(221)
            ax2.set_xlim([0,spacing*(shape[0]+1)])
            ax2.set_ylim([0,spacing*(shape[1]+1)])
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')           
            ax2.contourf(X,Y,Z_n0,cmap=cm.coolwarm,vmin=0)
            ax2.set_title('Real t=0')
            ax2.set_aspect(1)
            
            ax3 = plt.subplot(222)
            ax3.set_xlim([0,spacing*(shape[0]+1)])
            ax3.set_ylim([0,spacing*(shape[1]+1)])
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')         
            ax3.contourf(X,Y,Z_n1,cmap=cm.coolwarm,vmin=0)
            ax3.set_title('Real t=1')
            ax3.set_aspect(1)
            
            ax4 = plt.subplot(223)
            ax4.set_xlim([0,spacing*(shape[0]+1)])
            ax4.set_ylim([0,spacing*(shape[1]+1)])
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')         
            ax4.contourf(X,Y,Z_n0_PR,cmap=cm.coolwarm,vmin=0)
            ax4.set_title('Recon t=0')
            ax4.set_aspect(1)
            
            ax5 = plt.subplot(224)
            ax5.set_xlim([0,spacing*(shape[0]+1)])
            ax5.set_ylim([0,spacing*(shape[1]+1)])
            ax5.set_xlabel('x')
            ax5.set_ylabel('y')         
            ax5.contourf(X,Y,Z_n1_PR,cmap=cm.coolwarm,vmin=0)
            ax5.set_title('Recon t=1')
            ax5.set_aspect(1)
            plt.show(block=False)

 
    if Scaler:
        x_train = StdS_X.fit_transform(x_train)
        #y_train = StdS_Y.fit_transform(y_train)
    svr = SVR(kernel='linear',degree=5)

    print("training")
    svr.fit(x_train,y_train)
    
    #y_train_est = svr.predict(x_train)
    print(f"total training time: {time.time()-t0} s")
    
    #pickle.dump(svr,open(f'./numpy_arrays/svr_{data}_02i_2t.sav','wb'))
    #pickle.dump(svr,open(f'./Data_RT/SVR/{data_name}.sav','wb'))
    
    if save:
        if Scaler:
            pickle.dump(StdS_X,open(f'./Data_RT/SVR/{data_name}_StdS_X.sav','wb'))
            pickle.dump(svr,open(f'./Data_RT/SVR/{data_name}_StdS.sav','wb'))
            #pickle.dump(StdS_Y,open(f'./Data_RT/SVR/{data_name}_StdS_Y.sav','wb'))
        else:
            pickle.dump(svr,open(f'./Data_RT/SVR/{data_name}_1k.sav','wb'))
        
    #input()