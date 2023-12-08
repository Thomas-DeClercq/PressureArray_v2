#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:18:31 2021

@author: pc-robotiq
"""
import os
import numpy as np
import pandas as pd
import scipy
import math
from math import sqrt,pi,exp,sin,cos
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.patches as patches
import time
import random

from scipy.interpolate import Rbf
from rbf.interpolate import RBFInterpolant
#from rbf.interpolate import NearestRBFInterpolant

#from ROS_LocAndForceEstimation import HerzianContactLoc
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

def conv(x):
    return x.replace(',', '.').encode()


def compare2FEM(shape,spacing,I,name_file,type_contact):
    X = np.arange(0, (shape[0]+1)*spacing, 0.1)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    x = X.flatten()
    y = Y.flatten()
    
    #Z_calc = rotatedGuassian_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5])
    #Z_calc = roundedSurface_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5],par[6])
    pos = np.vstack((x,y))
    
    Z_calc_flatten = I(pos.T)
    Z_calc = scipy.interpolate.griddata((x,y), Z_calc_flatten, (X,Y), method='linear')
    
    
    raw_pressure = np.genfromtxt((conv(x) for x in open("/home/pc-robotiq/measurements_FEM/"+type_contact+"/raw_data/"+str(name_file)+"/Pressure.txt")),delimiter='\t',skip_header=1)
    raw_x_coord = np.genfromtxt((conv(x) for x in open("/home/pc-robotiq/measurements_FEM/"+type_contact+"/raw_data/"+str(name_file)+"/x_coord.txt")),delimiter='\t',skip_header=1)
    raw_y_coord = np.genfromtxt((conv(x) for x in open("/home/pc-robotiq/measurements_FEM/"+type_contact+"/raw_data/"+str(name_file)+"/y_coord.txt")),delimiter='\t',skip_header=1)

    all_data = np.zeros([len(raw_pressure),4])
    for i in range(len(raw_pressure)):
        all_data[i][0] = raw_pressure[i][0]
        all_data[i][3] = raw_pressure[i][1]*-1*10**6
        
        if all_data[i][0] == raw_x_coord[i][0]:
            all_data[i][1] = raw_x_coord[i][1]+13.5
        else:
            print("Mismatch in node numbering")
            
        if all_data[i][0] == raw_y_coord[i][0]:
            all_data[i][2] = raw_y_coord[i][1]+22.5
        else:
            print("Mismatch in node numbering")
            
    x = all_data.T[1]
    y = all_data.T[2]
    z = all_data.T[3]
            
    Z_meas = scipy.interpolate.griddata((x,y), z, (X,Y), method='linear')
    
    Z2 = Z_meas - Z_calc
    
    #fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)*10])
    #fig = plt.figure()
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.set_xlim([0,spacing*(shape[0]+1)])
    ax1.set_ylim([0,spacing*(shape[1]+1)])
    ax1.set_aspect('equal')
    
    CS = ax1.contourf(X,Y,Z_calc,cmap = cm.coolwarm,extend='both',vmin=0,vmax=Z_meas.max())
    
    #ax2 = fig.add_subplot(122)
    ax2.set_xlim([0,spacing*(shape[0]+1)])
    ax2.set_ylim([0,spacing*(shape[1]+1)])
    ax2.set_aspect('equal')
    
    ax2.contourf(X,Y,Z_meas,cmap = cm.coolwarm,extend='both',vmin=0,vmax=Z_meas.max())
    
    num_cir = shape[0]*shape[1]
    for i in range(num_cir):
        x = spacing+spacing*(i % shape[0])
        y = spacing+spacing*(i % shape[1])
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax1.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
        ax2.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
    
    cb_ax = fig.add_axes([0.92,0.1,0.02,0.8])
    cbar = fig.colorbar(CS,cax=cb_ax)
    
    sz = 10
    fig.set_size_inches(sz,sz/(shape[1])*shape[0])
    
    plt.show()
    return Z2
    
def generatePosition(shape,spacing):
    
    amount = int(shape[0]*shape[1])
     
    X = []
    Y = []
     
    for i in range(amount):
        x = spacing + spacing*(i % shape[0])
        X.append(x)
        y = spacing*shape[1] - (spacing*math.floor(i/shape[0]))
        Y.append(y)
        
    return np.array(X), np.array(Y)

#name_file = '-0,404226043915_-4,56785783664_173,430175862_17,3051882878_3,71580067543'
type_contact = 'LineContact'
name_dir = "/home/pc-robotiq/measurements_FEM/"+type_contact+"/csvs/"
list_files = os.listdir(name_dir)
list_errors = []
if 'old' in list_files:
    list_files.remove('old')
if '0_0_18_36_0.csv' in list_files:
    list_files.remove('0_0_18_36_0.csv')

for ind in [0]: #range(len(list_files)):
    #ind = 5

    #name_file = "/home/pc-robotiq/measurements_paper/FEM/csvs/"+name_file+".csv"
    name_file = list_files[ind]
    full_name = name_dir+name_file
    print("------------------------------------------------")
    print(ind+1)
    print(name_file)
    
    shape = [5,9]
    spacing = 4.5
    
    df=pd.read_csv(full_name, sep=',',header=None)
            
    data_raw = df.to_numpy()
    
    
    #data_raw = data_raw.T[random.randint(2,len(data_raw[0])-1)]
    data_raw = data_raw.T[-1]
    
    results = np.zeros((4,1))
    
    pres = data_raw[1:46].copy()
    x,y = generatePosition(shape,spacing)
    t0 = time.time()
    pos = np.zeros((2,len(x)))
    
    for i in range(len(x)):
        pos[0][i] = x[i]
        pos[1][i] = y[i]
    
    t0 = time.time()
    basis = 'ga' #phsX, mq, ga, exp, ...
    I = RBFInterpolant(pos.T, pres, sigma=100, phi=basis)
    print('duration: '+str(time.time()-t0))
    
    
    
    #errorMatrix = calc_error(shape,spacing,E.x,name_file[:-4],type_contact)
    errorMatrix = compare2FEM(shape,spacing,I,name_file[:-4],type_contact)
    print("RMSE: "+str(sqrt(np.square(errorMatrix).mean())))
    #plot_roundedSurface(shape,spacing,E.x)
    #F = calcForce_roundedSurface(E.x)
    #F_ref = data_raw[46]
    #print("reference force: "+str(F_ref))
    #print("force error: "+str(F-F_ref))
    
    list_errors.append(sqrt(np.square(errorMatrix).mean()))
    #time.sleep(15)
    
#print(":::::::::::::::::::::::::::::::::::")
#print(sum(list_errors)/len(list_errors))

"""
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([0,27])
ax1.set_ylim([0,45])
    
ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([0,27])
ax2.set_ylim([0,45])
"""