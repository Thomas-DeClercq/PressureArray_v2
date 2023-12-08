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
from scipy.optimize import Bounds, minimize, curve_fit
#from customScipy import curve_fit
from matplotlib import cm
import matplotlib.patches as patches
import time
import random

import nlopt


#from ROS_LocAndForceEstimation import HerzianContactLoc
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

def conv(x):
    return x.replace(',', '.').encode()

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

def getPressures(shape,spacing,par,plot=False):
    X = np.arange(0, (shape[0]+1)*spacing, 0.1)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    Z = roundedSurface_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5],par[6])
    
    pressures = np.array([])
    amount = shape[0]*shape[1]
            
    for i in range(amount):
        idx = i
        xxi = spacing + spacing*(idx % shape[0])
        yyi = spacing*shape[1] - (spacing*math.floor(idx/shape[0]))
        coord_idx = np.argwhere((X==xxi) & (Y==yyi))[0]
        p = Z[coord_idx[0]][coord_idx[1]]
        #print("["+str(xxi)+","+str(yyi)+"]: "+str(p))
        pressures = np.append(pressures,Z[coord_idx[0]][coord_idx[1]])
            
    if plot:
        fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
        ax = plt.subplot()
        ax.set_xlim([0,spacing*(shape[0]+1)])
        ax.set_ylim([0,spacing*(shape[1]+1)])
        
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
        
    #print(pressures)
    return pressures

def comparePressureDistributions(shape,spacing,par1,par2):
    X = np.arange(0, (shape[0]+1)*spacing, 0.1)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    Z_calc_1 = roundedSurface_array(X,Y,par1[0],par1[1],par1[2],par1[3],par1[4],par1[5],par1[6])
    Z_calc_2 = roundedSurface_array(X,Y,par2[0],par2[1],par2[2],par2[3],par2[4],par2[5],par2[6])
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.set_xlim([0,spacing*(shape[0]+1)])
    ax1.set_ylim([0,spacing*(shape[1]+1)])
    ax1.set_aspect('equal')
    ax1.set_title('Original Pressure Distribution')
    
    CS = ax1.contourf(X,Y,Z_calc_1,cmap = cm.coolwarm,extend='both',vmin=0,vmax=Z_calc_1.max())
    
    #ax2 = fig.add_subplot(122)
    ax2.set_xlim([0,spacing*(shape[0]+1)])
    ax2.set_ylim([0,spacing*(shape[1]+1)])
    ax2.set_aspect('equal')
    ax2.set_title('Reconstructed Pressure Distribution')
    
    CS2 = ax2.contourf(X,Y,Z_calc_2,cmap = cm.coolwarm,extend='both',vmin=0,vmax=Z_calc_1.max())
    
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
        
        CS = ax1.contourf(X,Y,Z_calc,cmap = cm.coolwarm,extend='both',vmin=0,vmax=Z_calc.max())
        
        #ax2 = fig.add_subplot(122)
        ax2.set_xlim([0,spacing*(shape[0]+1)])
        ax2.set_ylim([0,spacing*(shape[1]+1)])
        ax2.set_aspect('equal')
        
        CS2 = ax2.contourf(X,Y,Z_meas,cmap = cm.coolwarm,extend='both',vmin=0,vmax=Z_calc.max())
        
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

def RandomParams(shape,spacing):
    """
    lb = [x0[0], 0, -90, 0, 0, min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    ub = [x0[0]*10, 10, 90, spacing*max(shape), spacing*max(shape), max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    """
    p0 = random.uniform(10000,100000)
    std = random.uniform(0,10)
    theta = random.uniform(-90,90)
    lx = random.uniform(0,spacing*max(shape)/3)
    ly = random.uniform(0,spacing*max(shape)/3)
    x0 = random.uniform(0,spacing*(shape[0]+1))
    y0 = random.uniform(0,spacing*(shape[1]+1))
    return [p0,std,theta,lx,ly,x0,y0]

def roundedSurface_CF_2(X,p0,std,theta1,lx,ly,x0,y0):
    Z = np.zeros(X.shape[1])
    theta = theta1/180*pi
    
    X_theta = (X[0]-x0)*cos(theta)+(X[1]-y0)*sin(theta)
    Y_theta = -(X[0]-x0)*sin(theta)+(X[1]-y0)*cos(theta)
    
    #print(Z)
    for i, (x_theta, y_theta) in enumerate(zip(X_theta,Y_theta)):
        #x_theta = X_theta[i]
        #y_theta = Y_theta[i]
            
        if (abs(x_theta) < lx/2 and abs(y_theta) < ly/2):
            Z[i] = p0
        elif (abs(x_theta) > lx/2 and abs(y_theta) > ly/2):
            r = min((x_theta-lx/2)**2+(y_theta-ly/2)**2,
                    (x_theta+lx/2)**2+(y_theta-ly/2)**2,
                    (x_theta-lx/2)**2+(y_theta+ly/2)**2,
                    (x_theta+lx/2)**2+(y_theta+ly/2)**2)
            Z[i] = gaussian_r(sqrt(r),p0,std)
        elif x_theta < -lx/2:
            Z[i] = gaussian_r(-lx/2-x_theta,p0,std)
        elif x_theta > lx/2:
            Z[i] = gaussian_r(x_theta-lx/2,p0,std)
        elif y_theta < -ly/2:
            Z[i] = gaussian_r(-ly/2-y_theta,p0,std)
        elif y_theta > ly/2:
            Z[i] = gaussian_r(y_theta-ly/2,p0,std)
    return Z

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
    
    """
    bnds = scipy.optimize.Bounds([x0[0],0.01,-90.0,0.0,0.0,min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]),min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])],
                                  [float('inf'),float('inf'),90,float('inf'),float('inf'),max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]),max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])])
    
    """
    lb = [x0[0], 0, -90, 0, 0, min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    ub = [x0[0]*10, 10, 90, spacing*max(shape), spacing*max(shape), max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    
    
    #time0 = time.time()
    #E = minimize(ObjFun,x0,args=(array_n,n_p),method = meth[2],bounds=bnds,options=opt_2)
    xdata = array_n.T[2:].T[:n_p].T
    xdata = np.append(xdata,np.zeros((1,xdata.shape[1])),axis=0)
    ydata = array_n.T[1][:n_p]
    
    #print(xdata)
    #print(ydata)
    meth = ['trf','dogbox']
    
    E = curve_fit(roundedSurface_CF_2, xdata, ydata, x0, bounds=(lb, ub), method = meth[0], maxfev=5,verbose=2,ftol=1e-2)
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

if __name__ == '__main__':
    
    shape = [5,9]
    spacing = 4.5
    list_time = []
        
    for ind in range(1):#len(list_files)):
        print("------------------------------------------------")
        print(ind+1)
        shape = [5,9]
        spacing = 4.5
        
        params = RandomParams(shape,spacing)
        
        print(params)
        #plot pressure distribitution
        #plot_roundedSurface(shape,spacing,params)
        pres = getPressures(shape,spacing,params,plot=False)
    
        t0 = time.time()
                
        results = np.zeros((4,1))
    
        array, E = GaussianPressureDistribution(pres.copy(),shape,spacing,n=30)
        dur = time.time()-t0
        
        #print(E.x)
        comparePressureDistributions(shape,spacing,params,E.x)
        print('duration: '+str(dur))
        list_time.append(dur)
    
    
    
    print(":::::::::::::::::::::::::::::::::::")
    #print(sum(list_errors)/len(list_errors))
    list_time = np.array(list_time)
    print("mean duration: "+str(list_time.mean()))
