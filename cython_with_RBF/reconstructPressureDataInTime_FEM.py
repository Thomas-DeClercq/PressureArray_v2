#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:38:07 2022

@author: pc-robotiq
"""
import os
import pandas as pd
import numpy as np
import time
import scipy
from scipy.interpolate import RBFInterpolator, Rbf

from matplotlib import pyplot as plt
from matplotlib import cm, animation
import matplotlib.patches as patches

from PressureReconstruction import calc_Z, Optimization
from generatePressureDistributionsInTime import writeableData

def conv(x):
    return x.replace(',', '.').encode()

def FEM_data_contour(file_path):
    raw_pressure = np.genfromtxt((conv(x) for x in open(file_path+"/Pressure.txt")),delimiter='\t',skip_header=1)
    raw_x_coord = np.genfromtxt((conv(x) for x in open(file_path+"/x_coord.txt")),delimiter='\t',skip_header=1)
    raw_y_coord = np.genfromtxt((conv(x) for x in open(file_path+"/y_coord.txt")),delimiter='\t',skip_header=1)
    
    all_data = np.zeros([len(raw_pressure),4])
    for i in range(len(raw_pressure)):
        node = int(raw_pressure[i][0])
        all_data[i][0] = node
        all_data[i][3] = raw_pressure[i][1]*-1*10**6
        
        all_data[i][1] = raw_x_coord[node-1][1]+13.5
        all_data[i][2] = raw_y_coord[node-1][1]+22.5
    
    return all_data

def get_Pressure_array_FEM(file_dir):
    all_data_csv = pd.read_csv(file_dir+'.csv',header=None)
    all_data_np = all_data_csv.to_numpy()
    p = all_data_np.T[-1][1:46]
    return p

def calc_error(X,Y,Z_1,Z_2,Z_rbf):
    error = np.array([])
    Z_1f = Z_1.flatten()
    Z_2f = Z_2.flatten()
    for (z1,z2) in zip(Z_1f,Z_2f):
        error_i = (z1-z2)**2
        error = np.append(error,error_i)
        
    RMSE = (error.mean())**0.5/np.max(Z_1)*100
    #print('error pressure distribution: '+str(RMSE))
    
    error_rbf = np.array([])
    Z_rbff = Z_rbf.flatten()
    for (z1,zrbf) in zip(Z_1f,Z_rbff):
        error_j = (z1-zrbf)**2
        error_rbf = np.append(error_rbf,error_j)
        
    RMSE_rbf = (error_rbf.mean())**0.5/np.max(Z_1)*100
    #print('error rbf pressure distribution: '+str(RMSE_rbf))
    plt.show()
    return RMSE, RMSE_rbf

def animate(i,X,Y,all_params_in,all_Z_rbf,all_params_out):
    global cont1, cont2, cont3
    
    Z_in = np.zeros(X.shape)
    Z_out = np.zeros(X.shape)
    for itt in range(all_params_in.shape[1]):
        Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params_in[i][itt])
        Z_1 = np.reshape(Z_1,X.shape)
        Z_in = Z_in + Z_1
    
    for itt in range(all_params_out.shape[1]):
        if all_params_out[i][itt][0] != 0:
            Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params_out[i][itt])
            Z_1 = np.reshape(Z_1,X.shape)
            Z_out = Z_out + Z_1
    
    for c in cont1.collections:
        c.remove()
    for c in cont2.collections:
        c.remove()
    for c in cont3.collections:
        c.remove()
        
    cont1 = ax1.contourf(X,Y,Z_in,cmap=cm.coolwarm,vmin=0)
    cont2 = ax2.contourf(X,Y,all_Z_rbf[i],cmap=cm.coolwarm,vmin=0)
    cont3 = ax3.contourf(X,Y,Z_out,cmap=cm.coolwarm,vmin=0)
    TXT_t.set_text(str("t = ")+str(i))
    return cont1, cont2, cont3,

def getValue(X,Y,Z,xi,yi):
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    point = list(zip(xi,yi))
    #point = [(xi,yi)]
    zi = scipy.interpolate.griddata((X,Y),Z,point)
    zi = np.nan_to_num(zi)
    return zi

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
    plot_animation = False
    plot_end = True
    n = 50
    m = 10 # keep between 10 - 15
    p_m = 0.75
    max_it = 5
    max_t = 45
    
    error_rbf = np.array([])
    error_params = np.array([])
    dur_array = np.array([])
    
    xi = []
    yi = []
    
    X = np.linspace(0, (shape[0]+1)*spacing, n)
    Y = np.linspace(0, (shape[1]+1)*spacing, n)
    X, Y = np.meshgrid(X, Y)
        
    for x_i in range(shape[0]):
        for y_i in range(shape[1]):
            xi.append(spacing+x_i*spacing)
            yi.append(spacing+y_i*spacing)
            
    xi = np.array(xi)
    yi = np.array(yi)
    
    X_rbf = np.linspace(0, (shape[0]+1)*spacing, m)
    Y_rbf = np.linspace(0, (shape[1]+1)*spacing, m)
    X_rbf, Y_rbf = np.meshgrid(X_rbf, Y_rbf)

    name_dir = "./PressureDataInTime_FEM/C-shape/"
    list_files = os.listdir(name_dir)
    #for file in list_files:
    i=0
    while (i < len(list_files)):
        if '.' in list_files[i]:
            list_files.remove(list_files[i])
        else:
            i += 1
    
    #list_files.remove('.~lock.0_0.csv#')
    #name_file = "/home/pc-robotiq/measurements_paper/FEM/csvs/"+name_file+".csv"
    #for name in all_files:
    #list_files = ['0_0_8_1']
    for file in list_files:
        #file = all_files[0]
        file_dir = name_dir+file
        print(file)
        
        all_data = FEM_data_contour(file_dir)
        
        all_params_out = np.zeros((max_it,8))
        all_Z_rbf = []
        
        list_x0 = np.zeros((max_it,8))
        
        Z_in = np.zeros(X.shape)
        x = all_data.T[1]
        y = all_data.T[2]
        z = all_data.T[3]
        Z_in = scipy.interpolate.griddata((x,y), z, (X,Y), method='linear')
        
        Z_i = scipy.interpolate.griddata((X.flatten(),Y.flatten()),Z_in.flatten(),list(zip(xi,yi))) #get_Pressure_array_FEM(file_dir)#get Pressure array
        
        t0 = time.time()
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
        array, list_E = Optimization(xi,yi,Z_i.copy(),shape,spacing, array_rbf, list_x0, n=round(p_m*array_rbf.shape[1]),it_max=max_it,t_max=max_t)
        for idx,E in enumerate(list_E):
            list_x0[idx] = E.x
        te = time.time()
        dur = (te-t0)*10**3
        print("duration: "+str(dur))
        
        for itt in range(len(list_E)):
            all_params_out[itt] = list_E[itt].x
        
        Z_rbf_n = getValue(X_rbf, Y_rbf, Z_rbf, X.flatten(), Y.flatten())
        Z_rbf_n = Z_rbf_n.reshape(X.shape)
        all_Z_rbf.append(Z_rbf_n)
        
        Z_out = np.zeros(X.shape)
        for itt in range(all_params_out.shape[0]):
            if all_params_out[itt][0] != 0:
                Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params_out[itt])
                Z_1 = np.reshape(Z_1,X.shape)
                Z_out = Z_out + Z_1
        
        RMSE_i, RMSE_rbf_i = calc_error(X,Y,Z_in,Z_out,Z_rbf_n)
        error_rbf = np.append(RMSE_rbf_i,error_rbf)
        error_params = np.append(RMSE_i,error_params)
        dur_array = np.append(dur_array,dur)
        
        if plot_end:
            fig = plt.figure(figsize=[(shape[0]+1)*3,(shape[1]+1)])
            ax1 = plt.subplot(1,3,1)
            ax1.set_xlim([0,spacing*(shape[0]+1)])
            ax1.set_ylim([0,spacing*(shape[1]+1)])
            
            ax2 = plt.subplot(1,3,2)
            ax2.set_xlim([0,spacing*(shape[0]+1)])
            ax2.set_ylim([0,spacing*(shape[1]+1)])
            
            ax3 = plt.subplot(1,3,3)
            ax3.set_xlim([0,spacing*(shape[0]+1)])
            ax3.set_ylim([0,spacing*(shape[1]+1)])
            
            num_cir = int(shape[0]*shape[1])
            for i in range(num_cir):
                x = spacing+spacing*(i % shape[0])
                y = spacing+spacing*(i % shape[1])
                #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
                #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
                #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
                #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
                ax1.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
                ax2.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
                ax3.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
            
            x = all_data.T[1]
            y = all_data.T[2]
            z = all_data.T[3]
            
            #X = np.arange(0, (shape[0]+1)*spacing, 0.1)
            #Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
            #X, Y = np.meshgrid(X, Y)
            
            Z = scipy.interpolate.griddata((x,y), z, (X,Y), method='linear')
            
            max_pres = np.array(all_data).T[3].max()
            #print(max_pres)
            plt.contourf(X,Y,Z,cmap = cm.coolwarm,extend='both',vmin=0,vmax=max_pres)
            plt.title(file)
            
            num_cir = shape[0]*shape[1]
            for i in range(num_cir):
                    x = spacing+spacing*(i % shape[0])
                    y = spacing+spacing*(i % shape[1])
                    ax1.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
                    ax2.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
                    ax3.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
                    
            Z_out = np.zeros(X.shape)
            for itt in range(all_params_out.shape[0]):
                if all_params_out[itt][0] != 0:
                    Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params_out[itt])
                    Z_1 = np.reshape(Z_1,X.shape)
                    Z_out = Z_out + Z_1
                
            cont1 = ax1.contourf(X,Y,Z_in,cmap=cm.coolwarm,vmin=0)
            cont2 = ax2.contourf(X,Y,all_Z_rbf[0],cmap=cm.coolwarm,vmin=0)
            cont3 = ax3.contourf(X,Y,Z_out,cmap=cm.coolwarm,vmin=0)
        
    print('---------------------------------------------')
    print('error rbf: '+str(np.mean(error_rbf)))
    print('error params: '+str(np.mean(error_params)))
    print('average time: '+str(np.mean(dur_array)))
    print('max time: '+str(np.max(dur_array)))
    #df_out = writeableData(all_params_out)    
    
        
    if plot_animation:
        fig = plt.figure(figsize=[(shape[0]+1)*3,(shape[1]+1)])
        ax1 = plt.subplot(1,3,1)
        ax1.set_xlim([0,spacing*(shape[0]+1)])
        ax1.set_ylim([0,spacing*(shape[1]+1)])
        
        ax2 = plt.subplot(1,3,2)
        ax2.set_xlim([0,spacing*(shape[0]+1)])
        ax2.set_ylim([0,spacing*(shape[1]+1)])
        
        ax3 = plt.subplot(1,3,3)
        ax3.set_xlim([0,spacing*(shape[0]+1)])
        ax3.set_ylim([0,spacing*(shape[1]+1)])
        
        num_cir = int(shape[0]*shape[1])
        for i in range(num_cir):
            x = spacing+spacing*(i % shape[0])
            y = spacing+spacing*(i % shape[1])
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            ax1.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
            ax2.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
            ax3.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
            
        """
        Z_in = np.zeros(X.shape)
        for itt in range(all_params_in.shape[1]):
            Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params_in[itt])
            Z_1 = np.reshape(Z_1,X.shape)
            Z_in = Z_in + Z_1
        """
        Z_out = np.zeros(X.shape)
        for itt in range(all_params_out.shape[0]):
            if all_params_out[itt][0] != 0:
                Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params_out[itt])
                Z_1 = np.reshape(Z_1,X.shape)
                Z_out = Z_out + Z_1
            
        cont1 = ax1.contourf(X,Y,Z_in,cmap=cm.coolwarm,vmin=0)
        cont2 = ax2.contourf(X,Y,all_Z_rbf[0],cmap=cm.coolwarm,vmin=0)
        cont3 = ax3.contourf(X,Y,Z_out,cmap=cm.coolwarm,vmin=0)
        """
        TXT_t = plt.text(13.5,-4,str("t = 0"),size=15,horizontalalignment='center')
        
        ani = animation.FuncAnimation(fig,
                                      animate,
                                      fargs=(X,Y,all_params_in,all_Z_rbf,all_params_out),
                                      interval=1000,
                                      frames = timesteps,
                                      repeat=True,
                                      repeat_delay=0)
        
        #writervideo = animation.FFMpegWriter(fps=1)
        #ani.save("./PressureDataInTime_DEMOS/trans_rot.mp4",writer=writervideo)
        """
        
        
        
        
    
    
        
        