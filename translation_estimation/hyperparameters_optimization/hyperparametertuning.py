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

def plot_loss(history):
  fig1 = plt.figure()
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim(bottom=0)
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.savefig(f'./hyperparameters_optimization/plots/plot_loss_{hd}hd_{nodes}nodes_{lr}lr_{do}do.png')

def build_and_compile_model(norm,hidden_layers=2,npl=64,learning_rate=0.001,dropout=None):
  #dnn_model = build_and_compile_model(normalizer,hidden_layers=hd,npl=nodes,learning_rate=lr,dropout=do)
  if dropout is None:
      dropout = 0
  layers_list = [norm]
  for _ in range(hidden_layers):
      layers_list.append(layers.Dense(npl, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
      if do:
          layers_list.append(layers.Dropout(dropout))
  layers_list.append(layers.Dense(2))

  model = keras.Sequential(layers_list)

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate))
  return model

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
    save = True
    plot = True
    
    amount_of_hidden_layers = [1,2,3,4]
    nodes_per_layer = [32,64,128,512]
    options_learning_rate = [1e-2,1e-3,1e-4]
    options_dropout = [0.2,0.3,0.4,0.5]
    total_sims = 192 #= product of the lengths

    x_train = np.load('./hyperparameters_optimization/x_train.npy')
    y_train = np.load('./hyperparameters_optimization/y_train.npy')
    x_test = np.load('./hyperparameters_optimization/x_test.npy')
    y_test = np.load('./hyperparameters_optimization/y_test.npy')

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)    

    #start making and training the networks
    long_hd = []
    long_nodes = []
    long_lr = []
    long_do = []
    for hd in amount_of_hidden_layers:
        for nodes in nodes_per_layer:
            for lr in options_learning_rate:
                for do in options_dropout:
                    long_hd.append(hd)
                    long_nodes.append(nodes)
                    long_lr.append(lr)
                    long_do.append(do)

    c0 = 111 #start at this index
    long_hd = long_hd[c0:]
    long_nodes = long_nodes[c0:]
    long_lr = long_lr[c0:]
    long_do = long_do[c0:]

    counter_tot = len(long_hd)
    for counter in range(counter_tot):
        ci = counter+c0
        hd = long_hd[counter]
        nodes = long_nodes[counter]
        lr = long_lr[counter]
        do = long_do[counter]
        NN_name = f'{ci}_{hd}hd_{nodes}nodes_{str(lr).replace(".",",")}lr_{str(do).replace(".",",")}do'
        print(NN_name)

        t0 = time.time()
        dnn_model = build_and_compile_model(normalizer,hidden_layers=hd,npl=nodes,learning_rate=lr,dropout=do)
        dnn_model.summary()
        
        callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)]
        history = dnn_model.fit(
                        x_train,
                        y_train,
                        validation_split=0.2,
                        verbose=0, epochs=10000,
                        callbacks=callback)
        print(f'training time : {time.time()-t0}s')
        plot_loss(history)

        y_pred = dnn_model.predict(x_test)

        #print("error calculation")
        error_array_x = y_test.T[0].copy().flatten()
        error_array_x = np.vstack((error_array_x,y_pred.T[0].flatten()))
        error_array_x = np.vstack((error_array_x,abs(error_array_x[1] - error_array_x[0])))

        error_array_y = y_test.T[1].copy().flatten()
        error_array_y = np.vstack((error_array_y,y_pred.T[1].flatten()))
        error_array_y = np.vstack((error_array_y,abs(error_array_y[1] - error_array_y[0])))

        error_array = np.vstack((error_array_x,error_array_y))
        error_array = np.vstack((error_array,np.sqrt(error_array_x[2]**2+error_array_y[2]**2)))

        print(f"average x: {np.average(error_array_x[2])}")
        print(f"median x: {np.median(error_array_x[2])}")
        print(f"std x: {np.std(error_array_x[2])}")

        print(f"average y: {np.average(error_array_y[2])}")
        print(f"median y: {np.median(error_array_y[2])}")
        print(f"std y: {np.std(error_array_y[2])}")

        print(f"average: {np.average(error_array[-1])}")
        print(f"median: {np.median(error_array[-1])}")
        print(f"std: {np.std(error_array[-1])}")

        #save network and errors
        dnn_model.save(f'./hyperparameters_optimization/NNs/NN_{NN_name}')
        np.save(f'./hyperparameters_optimization/error_arrays/error_array_{NN_name}.npy',error_array)

        fig2 = plt.figure()
        ax2 = plt.subplot(411)
        ax2.scatter(error_array_x[0],error_array_x[1],s=1)
        ax2.set_xlabel('real x')
        ax2.set_ylabel('estimated x')
        ax2.plot([-spacing*shape[0],spacing*shape[0]],[-spacing*shape[0],spacing*shape[0]],color='k')
        ax2.set_xlim([-spacing*shape[0],spacing*shape[0]])
        ax2.set_ylim([-spacing*shape[0],spacing*shape[0]])

        ax3 = plt.subplot(412)
        ax3.scatter(error_array_y[0],error_array_y[1],s=1)
        ax3.set_xlabel('real y')
        ax3.set_ylabel('estimated y')
        ax3.plot([-spacing*shape[1],spacing*shape[1]],[-spacing*shape[1],spacing*shape[1]],color='k')
        ax3.set_xlim([-spacing*shape[1],spacing*shape[1]])
        ax3.set_ylim([-spacing*shape[1],spacing*shape[1]])

        ax4 = plt.subplot(413)
        ax4.scatter(np.sqrt(error_array_x[0]**2+error_array_y[0]**2),np.sqrt(error_array_x[1]**2+error_array_y[1]**2),s=1)
        ax4.set_xlabel('real dist')
        ax4.set_ylabel('estimated dist')
        ax4.plot([0,0],[-spacing*shape[1],spacing*shape[1]],color='k')
        ax4.set_xlim([0,spacing*shape[1]])
        ax4.set_ylim([0,spacing*shape[1]])

        ax5 = plt.subplot(414)
        ax5.boxplot([error_array_x[2],error_array_y[2],error_array[-1]], showfliers=False)
        ax5.set_ylim([0,10])
        #ax2.title(f'{hd}hd_{nodes}nodes_{lr}lr_{do}do')
        plt.savefig(f'./hyperparameters_optimization/plots/fit_{NN_name}.png')
        #plt.show(block=False)
        print('----------------------------------------------------------------')
        
        #time.sleep(1)
