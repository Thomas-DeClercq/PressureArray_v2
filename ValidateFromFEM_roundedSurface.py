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

#from ROS_LocAndForceEstimation import HerzianContactLoc
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

def conv(x):
    return x.replace(',', '.').encode()
    
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
    
def plot_roundedSurface(shape,spacing,par):
    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    
    X = np.arange(0, (shape[0]+1)*spacing, 0.1)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    Z = roundedSurface_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5],par[6]) 
    
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
    
    #Z_calc = rotatedGuassian_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5])
    Z_calc = roundedSurface_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5],par[6]) 
    
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

def compare2FEM(shape,spacing,par,name_file,type_contact,plot=True):
    X = np.arange(0, (shape[0]+1)*spacing, 0.1)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    #Z_calc = rotatedGuassian_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5])
    Z_calc = roundedSurface_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5],par[6]) 
    
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
    
    if plot:
        #fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)*10])
        #fig = plt.figure()
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.set_xlim([0,spacing*(shape[0]+1)])
        ax1.set_ylim([0,spacing*(shape[1]+1)])
        ax1.set_aspect('equal')
        ax1.set_title('Real pressure distribution (FEM)')
        
        CS = ax1.contourf(X,Y,Z_meas,cmap = cm.coolwarm,extend='both',vmin=0,vmax=Z_meas.max())
        
        #ax2 = fig.add_subplot(122)
        ax2.set_xlim([0,spacing*(shape[0]+1)])
        ax2.set_ylim([0,spacing*(shape[1]+1)])
        ax2.set_aspect('equal')
        ax2.set_title('Reconstructed pressure distribution')
        
        CS2 = ax2.contourf(X,Y,Z_calc,cmap = cm.coolwarm,extend='both',vmin=0,vmax=Z_meas.max())
        
        num_cir = shape[0]*shape[1]
        for i in range(num_cir):
            x = spacing+spacing*(i % shape[0])
            y = spacing+spacing*(i % shape[1])
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            ax1.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
            ax2.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))    
        
        cb_ax = fig.add_axes([0.92,0.1,0.02,0.8])
        cbar = fig.colorbar(CS,cax=cb_ax)
        #cbar.set_clim(vmin=0,vmax=Z_meas.max())
        
        sz = 10
        fig.set_size_inches(sz,sz/(shape[1])*shape[0])
    return Z2
    
def calcForce_roundedSurface(par):
    """
    par = [p0, std, theta, lx, ly, x0, y0] 
    """
    p0 = par[0]
    std = par[1]
    lx = par[3]
    ly = par[4]
    
    F_surface = p0*lx*ly*10**-6 #[N]
    F_lx = lx*p0*sqrt(2*pi)*std*10**-6
    F_ly = ly*p0*sqrt(2*pi)*std*10**-6
    F_c = p0*2*pi*std**2*10**-6
    return F_surface + F_lx + F_ly + F_c
    
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

def gaussian_r(r,p0,std):
    return p0*exp(-0.5*(r/std)**2)

def roundedSurface_array(X,Y,p0,std,theta,lx,ly,x0,y0):
    Z = np.zeros(X.shape)
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    X_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    Y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
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
            
            if (abs(x_theta) < lx/2 and abs(y_theta) < ly/2):
                Z[i][j] = p0
            elif (abs(x_theta) > lx/2 and abs(y_theta) > ly/2):
                r = min((x_theta-lx/2)**2+(y_theta-ly/2)**2,
                        (x_theta+lx/2)**2+(y_theta-ly/2)**2,
                        (x_theta-lx/2)**2+(y_theta+ly/2)**2,
                        (x_theta+lx/2)**2+(y_theta+ly/2)**2)
                Z[i][j] = gaussian_r(sqrt(r),p0,std)
            elif x_theta < -lx/2:
                Z[i][j] = gaussian_r(-lx/2-x_theta,p0,std)
            elif x_theta > lx/2:
                Z[i][j] = gaussian_r(x_theta-lx/2,p0,std)
            elif y_theta < -ly/2:
                Z[i][j] = gaussian_r(-ly/2-y_theta,p0,std)
            elif y_theta > ly/2:
                Z[i][j] = gaussian_r(y_theta-ly/2,p0,std)
    return Z

def roundedSurface(x,y,p0,std,theta,lx,ly,x0,y0):
    theta = theta/180*pi
    
    x_bar = x - x0
    y_bar = y - y0
    
    x_theta = x_bar*cos(theta) + y_bar*sin(theta)
    y_theta = -x_bar*sin(theta) + y_bar*cos(theta)
            
    if (abs(x_theta) < lx/2 and abs(y_theta) < ly/2):
        Z = p0
    elif (abs(x_theta) > lx/2 and abs(y_theta) > ly/2):
        r = min((x_theta-lx/2)**2+(y_theta-ly/2)**2,
                (x_theta+lx/2)**2+(y_theta-ly/2)**2,
                (x_theta-lx/2)**2+(y_theta+ly/2)**2,
                (x_theta+lx/2)**2+(y_theta+ly/2)**2)
        Z = gaussian_r(sqrt(r),p0,std)
    elif x_theta < -lx/2:
        Z = gaussian_r(-lx/2-x_theta,p0,std)
    elif x_theta > lx/2:
        Z = gaussian_r(x_theta-lx/2,p0,std)
    elif y_theta < -ly/2:
        Z = gaussian_r(-ly/2-y_theta,p0,std)
    elif y_theta > ly/2:
        Z = gaussian_r(y_theta-ly/2,p0,std)
    return Z


def ObjFun(par,array,n):
    #print(n)
    error = 0
    #print(array)
    for i in range(n):

        #sigma = rotatedGuassian(array[i][2],array[i][3],par[0],par[1],par[2],par[3],par[4],par[5])
        sigma = roundedSurface(array[i][2],array[i][3],par[0],par[1],par[2],par[3],par[4],par[5],par[6])
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
    length_0 = max(shape[0]*spacing,shape[1]*spacing)
    array_0 = np.zeros(array.shape)
    array_0.T[0] = array.T[0]
    array_0.T[1] = array.T[1].T/np.max(array.T[1])
    array_0.T[2] = array.T[2]/(length_0)
    array_0.T[3] = array.T[3]/(length_0)
    
    #print(array_0)
    n0 = 0
    for i0 in range(len(array_0)):
        if array_0[i0][1] > 0.4:
            n0 += 1
    
    #print(n0)
    diff_x = False
    diff_y = False
    
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
    
    # x0 = [p0, std, theta, lx, ly, x0, y0]
    
    if n0 >= 5:
        x0 = [array_n[0][1],2,0,2,2,(array_n.T[2][nx[0]]+array_n.T[2][nx[1]])/2,(array_n.T[3][ny[0]]+array_n.T[3][ny[1]])/2]
    else:
        x0 = [array_n[0][1],3,0,0,0,(array_n.T[2][nx[0]]+array_n.T[2][nx[1]])/2,(array_n.T[3][ny[0]]+array_n.T[3][ny[1]])/2]
    #x0 = [array_n[0][1],0.1,array_n[0][2],array_n[0][3]]
    
    #print(x0)
    
    bnds = scipy.optimize.Bounds([x0[0],0.01,-90.0,0.0,0.0,min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]),min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])],
                                  [float('inf'),float('inf'),90,float('inf'),float('inf'),max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]),max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])])
    
    #print(bnds)
    meth = ['Nelder-Mead','L-BFGS-B','Powell','TNC','SLSQP']
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    opt = {}
    opt_0 = {'maxfev':10**4}
    opt_1 = {'ftol':1e-9,'gtol':1e-7,'maxls':40}
    opt_2 = {'ftol':1e-7, 'ftol':1e-7}
    opt_4 = {'ftol':1e-12,'maxiter':2*10**4}
    
    E = minimize(ObjFun,x0,args=(array_n,n_p),method = meth[2],bounds=bnds,options=opt)
    
    #print(E)
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

list_errors = np.array([])
list_errors_P = np.array([])
#name_file = '-0,404226043915_-4,56785783664_173,430175862_17,3051882878_3,71580067543'
type_contacts = ['PointContact','LineContact','SurfaceContact']

for type_contact in type_contacts:
    name_dir = "/home/pc-robotiq/measurements_FEM/"+type_contact+"/csvs/"
    list_files = os.listdir(name_dir)
    
    if 'old' in list_files:
        list_files.remove('old')
    if '0_0_18_36_0.csv' in list_files:
        list_files.remove('0_0_18_36_0.csv')
    if '.~lock.-0,419031659103_6,70554571632_17,1619366818_15,1065354795_226,146914189.csv#' in list_files:
        list_files.remove('.~lock.-0,419031659103_6,70554571632_17,1619366818_15,1065354795_226,146914189.csv#')
        
    
    for ind in range(len(list_files)):
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
                
        data_raw_all = df.to_numpy()
        list_error_i = []
        
        for ti in range(1):
        #data_raw = data_raw.T[random.randint(2,len(data_raw[0])-1)]
        
            #data_raw = data_raw_all.T[ti]
            data_raw = data_raw_all.T[-1]
                    
            results = np.zeros((4,1))
            
            pres = data_raw[1:46].copy()
            t0 = time.time()
            E = GaussianPressureDistribution(pres.copy(),shape,spacing,n=30)[1]
            print('duration: '+str(time.time()-t0))
            
            #errorMatrix = calc_error(shape,spacing,E.x,name_file[:-4],type_contact)
            errorMatrix = compare2FEM(shape,spacing,E.x,name_file[:-4],type_contact,plot=True)
            #sprint("RMSE: "+str(sqrt(np.square(errorMatrix).mean())))
            #plot_roundedSurface(shape,spacing,E.x)
            F = calcForce_roundedSurface(E.x)
            F_ref = data_raw[46]
            print("reference force: "+str(F_ref))
            print("force error: "+str(F-F_ref))
            
            
            #list_error_i.append(sqrt(np.square(errorMatrix).mean()))
            list_error_i.append((F-F_ref)/F_ref)
            list_errors_P = np.append(list_errors_P,sqrt(np.square(errorMatrix).mean()))
        
        if len(list_errors) == 0:
            list_errors = np.append(list_errors,list_error_i)
        else:
            list_errors = np.vstack((list_errors,list_error_i))
        #time.sleep(15)

fig1 = plt.figure(figsize=[12,8])
ax1 = plt.subplot()
ax1.boxplot(list_errors.flatten(),showfliers=True)

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