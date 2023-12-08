import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import time, random, os

import scipy
from scipy.interpolate import RBFInterpolator, Rbf

import sys
sys.path.insert(1,'/home/thomas/pythonScripts/PressureArray_v2/PR_cython')
from PressureReconstruction_update210623 import calc_Z, Optimization

def getValue(X,Y,Z,xi,yi):
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    point = list(zip(xi,yi))
    #point = [(xi,yi)]
    zi = scipy.interpolate.griddata((X,Y),Z,point)
    zi = np.nan_to_num(zi)
    return zi
def pressureReconstruction_df(df,shape,spacing,n=50,m=10,p_m=0.75,iterations=5):
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
    it_out = iterations
    
    list_x0 = np.zeros((it_out,8))
    df_new = np.zeros((timesteps*it_out,12))
    
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
            df_new[t*it_out+idx1,10] = df.loc[df['Timestep']==t].values[-1][-2]
            df_new[t*it_out+idx1,11] = df.loc[df['Timestep']==t].values[-1][-1]
            for idx2 in range(8):
                df_new[t*it_out+idx1,2+idx2] = list_x0[idx1,idx2]
                
    columns = ['Timestep','iteration','p0','std','lx','ly','r_curve','theta','x0','y0','Dx','Dy']
    df_new = pd.DataFrame(df_new,columns=columns)        
    return df_new

if __name__ == "__main__":
    t0 = time.time()
    shape = [4,8]
    spacing = 4.5
    plot_contour = False
    n = 50
    dir_data = "/home/thomas/pythonScripts/PressureArray_v2/translation_estimation/Data/"
    training_data = 'TrainingData/only_trans'
    test_data = 'TestData/only_trans'
    save = True
    plot = True

    PR_it = 1
    training_size = 10000
    
    all_training_files = os.listdir(f"{dir_data}/{training_data}")
    random.shuffle(all_training_files)
    training_files = all_training_files[:training_size]

    test_files = os.listdir(f"{dir_data}/{test_data}")
    #all_validation_files = os.listdir(f"{dir_data}/{test_data}")
    
    #error_array = np.zeros((len(all_files),2))
    #file = all_files[10]
    
    time_steps = 1 #currently max of 1, can later be increased --> doens't improve accuracy, training time x2
    
    amount_of_hidden_layers = [1,2,3,4]
    nodes_per_layer = [32,64,128,512]
    options_learning_rate = [1e-2,1e-3,1e-4]
    options_dropout = [0.2,0.3,0.4,0.5]
    total_sims = 192#= product of the lengths

    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    
    t0 = time.time()
    progress_counter = 0.1
    for idx,file in enumerate(training_files):
        if (idx/len(training_files)) >= progress_counter:
            print(f'{round(progress_counter*100)}% done')
            progress_counter = progress_counter + 0.1
        #df = pd.read_csv("./"+data+"/"+file)
        df = pd.read_csv(f"{dir_data}/{training_data}/{file}")
        
        df = pressureReconstruction_df(df,shape,spacing,iterations=PR_it)
        
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
                x_train_new = df_t.values.T[2:10].T.flatten()
                x_train = np.vstack((x_train,x_train_new))
                y_train_i = df_t.loc[df["Timestep"]==current_timestep][['Dx','Dy']].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_train = np.vstack((y_train,y_train_i))

            except: #only for the first time ever
                current_timestep = df_t["Timestep"].max()
                x_train_new = df_t.values.T[2:10].T.flatten()
                x_train = np.append(x_train,x_train_new)
                y_train_i = df_t.loc[df["Timestep"]==current_timestep][['Dx','Dy']].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_train = np.append(y_train,y_train_i)
    
    print('Training data converted')

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)
    #print(normalizer.mean.numpy())

    progress_counter = 0.1
    for idx,file in enumerate(test_files):
        if (idx/len(test_files)) >= progress_counter:
            print(f'{round(progress_counter*100)}% done')
            progress_counter = progress_counter + 0.1
        #df = pd.read_csv("./"+data+"/"+file)
        df = pd.read_csv(f"{dir_data}/{test_data}/{file}")
        
        df = pressureReconstruction_df(df,shape,spacing,iterations=PR_it)
        
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
                y_test_i = df_t.loc[df["Timestep"]==current_timestep][['Dx','Dy']].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_test = np.vstack((y_test,y_test_i))

            except: #only for the first time ever
                current_timestep = df_t["Timestep"].max()
                x_test_new = df_t.values.T[2:10].T.flatten()
                x_test = np.append(x_test,x_test_new)
                y_test_i = df_t.loc[df["Timestep"]==current_timestep][['Dx','Dy']].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_test = np.append(y_test,y_test_i)

    print('Test data converted')

    np.save('./hyperparameters_optimization/x_train.npy',x_train)
    np.save('./hyperparameters_optimization/y_train.npy',y_train)
    np.save('./hyperparameters_optimization/x_test.npy',x_test)
    np.save('./hyperparameters_optimization/y_test.npy',y_test)