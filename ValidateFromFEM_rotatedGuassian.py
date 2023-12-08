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

#from ROS_LocAndForceEstimation import HerzianContactLoc
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

def conv(x):
    return x.replace(',', '.').encode()

def plot_gaussian_2d(shape,spacing,par):    
    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    
    X = np.arange(0, (shape[0]+1)*spacing, 0.1)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    Z = gaussian_2d_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5])
    
    plt.contourf(X,Y,Z,cmap = cm.coolwarm,extend='both',vmin=0,vmax=par[0])
    
    num_cir = shape[0]*shape[1]
    for i in range(num_cir):
        x = spacing+spacing*(i % shape[0])
        y = spacing+spacing*(i % shape[1])
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
    
    plt.show()
    
def plot_rotatedGaussian(shape,spacing,par):
    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    
    X = np.arange(0, (shape[0]+1)*spacing, 0.1)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    Z = rotatedGuassian_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5]) 
    
    plt.contourf(X,Y,Z,cmap = cm.coolwarm,extend='both',vmin=0,vmax=par[0])
    plt.colorbar()
    
    num_cir = shape[0]*shape[1]
    for i in range(num_cir):
        x = spacing+spacing*(i % shape[0])
        y = spacing+spacing*(i % shape[1])
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
    
    plt.show()
    
def calc_error(shape,spacing,par,name_file,type_contact):
    X = np.arange(0, (shape[0]+1)*spacing, 0.1)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    Z_calc = rotatedGuassian_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5]) 
    
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
    return Z2
    
def butter_lowpass_filter(data,on):
    if on:
        fs = 30
        cutoff = 5
        nyq = 0.5*fs
        order = 2
        
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff,btype='low', analog=False)
        # y = filtfilt(b,a,data)
        
        y = scipy.signal.lfilter(b,a,data)
    else:
        y = data
    return y

def gaussian(x,y,p0,std,x0,y0):
    return p0*exp(-0.5*(((x-x0)**2+(y-y0)**2)/(std**2)))

def gaussian_2d(x,y,p0,std_x,std_y,rho,x0,y0):
    return p0*exp(-0.5/(1-rho**2)*(((x-x0)/std_x)**2-2*rho*((x-x0)/std_x)*((y-y0)/std_y)+((y-y0)/std_y)**2))

def gaussian_2d_array(x,y,p0,std_x,std_y,rho,x0,y0):
    return p0*np.exp(-0.5/(1-rho**2)*(((x-x0)/std_x)**2-2*rho*((x-x0)/std_x)*((y-y0)/std_y)+((y-y0)/std_y)**2))

def rotatedGuassian_array(X,Y,p0,std_x,std_y,theta,x0,y0):
    Z = np.zeros(X.shape)
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    X_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    Y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
    rho = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            """
            x = X[i][j]
            y = Y[i][j]
            x_bar = x - x0
            y_bar = y - y0
            """
            x_theta = X_theta[i][j]
            y_theta = Y_theta[i][j]
            
            Z[i][j] = p0*exp(-0.5*(((x_theta)/std_x)**2+((y_theta)/std_y)**2))
    return Z

def rotatedGuassian(X,Y,p0,std_x,std_y,theta,x0,y0):
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    x_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
    rho = 0
            
    Z = p0*exp(-0.5*(((x_theta)/std_x)**2+((y_theta)/std_y)**2))
    return Z

def ObjFun(par,array,n):
    #print(n)
    error = 0
    #print(array)
    for i in range(n):
        #sigma = gaussian(array[i][2],array[i][3],par[0],par[1],par[2],par[3])
        
        #sigma = gaussian_2d(array[i][2],array[i][3],par[0],par[1],par[2],par[3],par[4],par[5])
        sigma = rotatedGuassian(array[i][2],array[i][3],par[0],par[1],par[2],par[3],par[4],par[5])
        #sigma = gaussian(array[i][2],array[i][3],par[0],2,par[2],par[3])
        #print(sigma)
        #error += (len(array)-i)**2*abs(sigma-array[i][1])
        #print(abs(sigma-array[i][1]))
        #error += 1/array[i][1]*abs(sigma-array[i][1])
        error += abs(sigma-array[i][1])
        
    #error += (1/par[1])*10**-1.5
    #print(error)
    #print(par)
    #print('-----------------------------------')
    
        
    return error

def GaussianPressureDistribution(pressureArray, shape, spacing, t=6,n=9):
    """
    n: amount of points used
    """
    global x0
    n_p = n
    
    # t0 = time.time()
    amount = int(shape[0]*shape[1])
    
    array = np.zeros((amount,4)) # n points with highest pressure; 1: sensor number; 2: pressure; 3: x coordinate; 4: y coordinate
    
    for i in range(amount):
        array[i][0] = np.argmax(pressureArray)
        array[i][1] = pressureArray[int(array[i][0])]
        
        pressureArray[int(array[i][0])] = -10**9
        
    for i in range(amount):
        idx = array[i][0]
        x = spacing + spacing*(idx % shape[0])
        y = spacing*shape[1] - (spacing*math.floor(idx/shape[0]))
        array[i][2] = x
        array[i][3] = y
         
    #print(array)    
    
    #x0 = [array[0][1],2,array[0][2],array[0][3]] # [k, a, x_c, y_c]
    #bnds = scipy.optimize.Bounds([array[0][1],0,array[0][2]-spacing,array[0][3]-spacing],[float('inf'),float('inf'),array[0][2]-spacing,array[0][3]-spacing])
    """
    if not success:
        x0 = [array[0][1],1,array[0][2],array[0][3]] # [k, x_c, y_c]
    """
    #x0 = [array[0][1]/gaussian(0,0,1,1,0,0),1,array[0][2],array[0][3]]
    #bnds = scipy.optimize.Bounds([array[0][1]//gaussian(0,0,1,1,0,0),0,min(array[0][2]-spacing,0),min(array[0][3]-spacing,0)],[float('inf'),float('inf'),max(array[0][2]+spacing,shape[0]*spacing),max(array[0][3]+spacing,shape[0]*spacing)])
    
    
    ### normalize
    length_n = max(shape[0]*spacing,shape[1]*spacing)
    array_n = np.zeros(array.shape)
    array_n.T[0] = array.T[0]
    array_n.T[1] = array.T[1].T/np.max(array.T[1])
    array_n.T[2] = array.T[2]/(length_n)
    array_n.T[3] = array.T[3]/(length_n)
    
    #print(array_n)
    """
    if not success:
        x0 = [array_n[0][1],0.1,array_n[0][2],array_n[0][3]]
        
    else:
        x0[0] = x0[0]/np.max(array.T[1])
        x0[1] = 0.1
        x0[2] = x0[2]/(length_n)
        x0[3] = x0[3]/(length_n)
    """
    

    
    ## have to adjust the bounds due to new normalization constant length_n
    #bnds = scipy.optimize.Bounds([x0[0],0,max(array_n[0][2]-1/shape[0],0),max(array_n[0][3]-1/shape[1],0)],[float('inf'),float('inf'),min(array_n[0][2]+1/shape[0],1),min(array_n[0][3]+1/shape[1],1)])   
    #bnds = scipy.optimize.Bounds([x0[0],0,max(array_n[0][2]-1/(2*shape[0]),0),max(array_n[0][3]-1/(2*shape[1]),0)],[float('inf'),float('inf'),min(array_n[0][2]+1/(2*shape[0]),1),min(array_n[0][3]+1/(2*shape[1]),1)])   
    #bnds = scipy.optimize.Bounds([x0[0],0,max(array_n[0][2]-1/(2*shape[0]),1/(2*shape[0])),max(array_n[0][3]-1/(2*shape[1]),1/(2*shape[1]))],[float('inf'),float('inf'),min(array_n[0][2]+1/(2*shape[0]),1-1/(2*shape[0])),min(array_n[0][3]+1/(2*shape[1]),1-1/(2*shape[1]))])   
    #bnds = scipy.optimize.Bounds([x0[0],0,max(array_n[0][2]-1/(shape[0]),1/(2*shape[0])),max(array_n[0][3]-1/(shape[1]),1/(2*shape[1]))],[float('inf'),float('inf'),min(array_n[0][2]+1/(shape[0]),1-1/(2*shape[0])),min(array_n[0][3]+1/(shape[1]),1-1/(2*shape[1]))])   

    diff_x = False
    diff_y = False
    """
    for i in range(n):
        for j in range(i):
            if array[i][2] != array[j][2]:
                diff_x = True
            if array[i][3] != array[j][3]:
                diff_y = True
                
    if diff_x and diff_y:
        idx = n
    else:
        idx = n + 1
    """
    
    array_n = array
    
    n = 3
    nx = []
    ny = []
    while not (diff_x and diff_y):
        for i in range(n):
            for j in range(i):
                if (array[i][2] != array[j][2]) and (not diff_x):
                    diff_x = True
                    nx.append(j)
                    nx.append(i)
                if (array[i][3] != array[j][3]) and (not diff_y):
                    diff_y = True
                    ny.append(j)
                    ny.append(i)
        if not (diff_x and diff_y):
            n += 1
    
    x0 = [array_n[0][1],1,1,0,(array_n.T[2][nx[0]]+array_n.T[2][nx[1]])/2,(array_n.T[3][ny[0]]+array_n.T[3][ny[1]])/2]
    #x0 = [array_n[0][1],0.1,array_n[0][2],array_n[0][3]]
    
    print(x0)
    
    bnds = scipy.optimize.Bounds([x0[0],0.0,0.0,-45.0,min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]),min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])],[float('inf'),float('inf'),float('inf'),45,max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]),max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])])
    print(bnds)
    meth = ['Nelder-Mead','L-BFGS-B','Powell','TNC','SLSQP']
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    opt = {}
    opt_0 = {'maxfev':10**4}
    opt_1 = {'ftol':1e-9,'gtol':1e-7,'maxls':40}
    opt_2 = {'ftol':1e-7, 'ftol':1e-7}
    opt_4 = {'ftol':1e-12,'maxiter':2*10**4}
    
    E = minimize(ObjFun,x0,args=(array_n,n_p),method = meth[1],bounds=bnds,options=opt_1)
    
    print(E)
    ### denormalize
    #E.x[0] = E.x[0]*np.max(array.T[1])
    #E.x[4] = E.x[4]*length_n
    #E.x[5] = E.x[5]*length_n
    """
    if E.success:
        x0 = E.x
    """
    #E.x[0] = E.x[0]/(E.x[1])**2 #==> E.x = [p0,z,x_c,y_c] p0 in Pa
    #print(E)
    #print(1/(time.time()-t0))
    return array,E

#name_file = '-0,404226043915_-4,56785783664_173,430175862_17,3051882878_3,71580067543'
type_contact = 'LineContact'
ind = 2
name_dir = "/home/pc-robotiq/measurements_FEM/"+type_contact+"/csvs/"
list_files = os.listdir(name_dir)
#list_files.remove('.~lock.0_0.csv#')
list_files.remove('old')
#name_file = "/home/pc-robotiq/measurements_paper/FEM/csvs/"+name_file+".csv"
name_file = list_files[ind]
full_name = name_dir+name_file
print(name_file)

shape = [5,9]
spacing = 4.5

df=pd.read_csv(full_name, sep=',',header=None)
        
data_raw = df.to_numpy()
        
data_raw = data_raw.T[-1]

results = np.zeros((4,1))

pres = data_raw[1:46].copy()
t0 = time.time()
E = GaussianPressureDistribution(pres.copy(),shape,spacing,n=30)[1]
print('duration: '+str(time.time()-t0))

errorMatrix = calc_error(shape,spacing,E.x,name_file[:-4],type_contact)
print(sqrt(np.square(errorMatrix).mean()))
plot_rotatedGaussian(shape,spacing,E.x)
"""
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([0,27])
ax1.set_ylim([0,45])
    
ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([0,27])
ax2.set_ylim([0,45])
"""