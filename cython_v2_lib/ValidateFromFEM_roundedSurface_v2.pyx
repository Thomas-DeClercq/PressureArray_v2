#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:18:31 2021

@author: pc-robotiq
"""
cimport cython


import os
import numpy as np
cimport numpy as np

import pandas as pd
import scipy
from scipy.optimize import curve_fit
from matplotlib import cm
import matplotlib.patches as patches
import time

from scipy.integrate import simps

import math
from libc.math cimport sqrt,pi,exp,sin,cos,atan2
from libc.math cimport fabs, fmin

ctypedef np.double_t DTYPE_t
from cpython cimport array

from multiprocessing import Process, Value, Array
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

#from ROS_LocAndForceEstimation import HerzianContactLoc
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

def conv(x):
    return x.replace(',', '.').encode()
    
def compare2FEM(shape,spacing,par,name_file,type_contact,plot=True):
    X = np.arange(0, (shape[0]+1)*spacing, 0.1)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    #Z_calc = rotatedGuassian_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5])
    Z_calc = roundedSurface_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5],par[6],par[7]) 
    
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

def roundedSurface_array_backup(X,Y, p0, std, theta, lx, ly, x0, y0, ttype):
    int_type = round(ttype)
    
    Z = np.zeros(X.shape)
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    X_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    Y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
    
    if int_type == 1:
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
    elif int_type == 2:
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
                
                if ((x_theta/lx)**2+(y_theta/ly)**2) <= 1:
                    Z[i][j] = p0
                else:
                    alpha = np.arctan2(y_theta,x_theta)
                    r_e = (lx*ly)/sqrt(lx**2*sin(alpha)**2+ly**2*cos(alpha)**2)
                    r = sqrt(x_theta**2+y_theta**2)-r_e
                    Z[i][j] = gaussian_r(r,p0,std)
    return Z

cdef roundedSurface_array(np.ndarray[DTYPE_t,ndim=2] X,np.ndarray[DTYPE_t,ndim=2] Y, float p0, float std, float theta1, float lx, float ly, float x0, float y0, float ttype):
    cdef int int_type = round(ttype)
    
    cdef np.ndarray[DTYPE_t,ndim=2] Z = np.zeros((X.shape[0],X.shape[1]))
    cdef float theta = theta1/180*pi

    cdef np.ndarray[DTYPE_t,ndim=2] X_theta = (X-x0)*cos(theta)+(Y-y0)*sin(theta)
    cdef np.ndarray[DTYPE_t,ndim=2] Y_theta = -(X-x0)*sin(theta)+(Y-y0)*cos(theta)
    
    cdef Py_ssize_t i, j
    cdef double x_theta,y_theta
    cdef float r, r_e, alpha
    if int_type == 1:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                """
                x = X[i][j]
                y = Y[i][j]
                x_bar = x - x0
                y_bar = y - y0
                """
                x_theta = X_theta[i,j]
                y_theta = Y_theta[i,j]
                
                if (abs(x_theta) < lx/2 and abs(y_theta) < ly/2):
                    Z[i,j] = p0
                elif (abs(x_theta) > lx/2 and abs(y_theta) > ly/2):
                    r = min((x_theta-lx/2)**2+(y_theta-ly/2)**2,
                            (x_theta+lx/2)**2+(y_theta-ly/2)**2,
                            (x_theta-lx/2)**2+(y_theta+ly/2)**2,
                            (x_theta+lx/2)**2+(y_theta+ly/2)**2)
                    Z[i,j] = gaussian_r(sqrt(r),p0,std)
                elif x_theta < -lx/2:
                    Z[i,j] = gaussian_r(-lx/2-x_theta,p0,std)
                elif x_theta > lx/2:
                    Z[i,j] = gaussian_r(x_theta-lx/2,p0,std)
                elif y_theta < -ly/2:
                    Z[i,j] = gaussian_r(-ly/2-y_theta,p0,std)
                elif y_theta > ly/2:
                    Z[i,j] = gaussian_r(y_theta-ly/2,p0,std)
    elif int_type == 2:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                """
                x = X[i][j]
                y = Y[i][j]
                x_bar = x - x0
                y_bar = y - y0
                """
                x_theta = X_theta[i,j]
                y_theta = Y_theta[i,j]
                
                if ((x_theta/lx)**2+(y_theta/ly)**2) <= 1:
                    Z[i,j] = p0
                else:
                    alpha = np.arctan2(y_theta,x_theta)
                    r_e = (lx*ly)/sqrt(lx**2*sin(alpha)**2+ly**2*cos(alpha)**2)
                    r = sqrt(x_theta**2+y_theta**2)-r_e
                    Z[i,j] = gaussian_r(r,p0,std)
    #print(Z)
    return Z

def calcForce_int(par, shape, spacing, n=20):
    """
    par = [p0, std, theta, lx, ly, x0, y0, type] 
    """
    #t0 = time.time()
    x = np.linspace(0, (shape[0]+1)*spacing, n)
    y = np.linspace(0, (shape[1]+1)*spacing, n)
    X, Y = np.meshgrid(x, y)
    #print(X)
    #t1 = time.time()
    Z_calc = roundedSurface_array(X,Y,par[0],par[1],par[2],par[3],par[4],par[5],par[6],round(par[7]))
    #print(Z_calc)
    #t2 = time.time()
    F = simps([simps(zz_x, x) for zz_x in Z_calc], y)
    #t3 = time.time()
    #print('t0-t1: '+str(t1-t0))
    #print('t1-t2: '+str(t2-t1))
    #print('t2-t3: '+str(t3-t2))
    return F*1e-6

cdef double gaussian_r(double r, double p0, double std):
    return p0*exp(-0.5*(r/std)**2)

cdef make_feasible(x0,lb,ub):
    for i,(x,lv, uv) in enumerate(zip(x0,lb,ub)):
        if x < lv:
            x0[i] = lv
        elif x > uv:
            x0[i] = uv
    return x0

def roundedSurface_CF_rect(np.ndarray[DTYPE_t,ndim=2] X, double p0, double std, double theta1, double lx, double ly, double x0, double y0):
    #nog opt
    cdef double theta = theta1/180*pi
    
    cdef np.ndarray[DTYPE_t] X_theta = (X[0]-x0)*cos(theta)+(X[1]-y0)*sin(theta)
    cdef np.ndarray[DTYPE_t] Y_theta = -(X[0]-x0)*sin(theta)+(X[1]-y0)*cos(theta)
    
    #cdef np.ndarray[DTYPE_t] Z = np.zeros(len(X_theta))
    
    for i in range(len(X_theta)):
        x_theta = X_theta[i]
        y_theta = Y_theta[i]
        
        if (fabs(x_theta) < lx/2 and fabs(y_theta) < ly/2):
            X[2][i] = p0
        elif (fabs(x_theta) > lx/2 and fabs(y_theta) > ly/2):
            r = fmin((x_theta-lx/2)**2+(y_theta-ly/2)**2,
                fmin((x_theta+lx/2)**2+(y_theta-ly/2)**2,
                fmin((x_theta-lx/2)**2+(y_theta+ly/2)**2,
                (x_theta+lx/2)**2+(y_theta+ly/2)**2)))
            r = sqrt(r)
            X[2][i] = p0*exp(-0.5*(r/std)**2)
        elif x_theta < -lx/2:
            r = -lx/2-x_theta
            X[2][i] = p0*exp(-0.5*(r/std)**2)
        elif x_theta > lx/2:
            r = x_theta-lx/2
            X[2][i] = p0*exp(-0.5*(r/std)**2)
        elif y_theta < -ly/2:
            r = -ly/2-y_theta
            X[2][i] = p0*exp(-0.5*(r/std)**2)
        elif y_theta > ly/2:
            r = y_theta - ly/2
            X[2][i] = p0*exp(-0.5*(r/std)**2)
            
    #print(X[2])
    return X[2]

def roundedSurface_CF_ellips(np.ndarray[DTYPE_t,ndim=2] X, double p0, double std, double theta1, double lx, double ly, double x0, double y0):
    #nog opt^
    cdef double theta = theta1/180*pi
    
    cdef np.ndarray[DTYPE_t] X_theta = (X[0]-x0)*cos(theta)+(X[1]-y0)*sin(theta)
    cdef np.ndarray[DTYPE_t] Y_theta = -(X[0]-x0)*sin(theta)+(X[1]-y0)*cos(theta)
    
    cdef double x_theta, y_theta, r, alpha, r_e
    cdef int i

    for i in range(len(X_theta)):
        x_theta = X_theta[i]
        y_theta = Y_theta[i]
                
        if ((x_theta/lx)**2+(y_theta/ly)**2) <= 1:
            X[2][i] = p0
        else:
            alpha = atan2(y_theta,x_theta)
            r_e = (lx*ly)/sqrt(lx**2*sin(alpha)**2+ly**2*cos(alpha)**2)
            r = sqrt(x_theta**2+y_theta**2)-r_e
            X[2][i] = p0*exp(-0.5*(r/std)**2)
            
    return X[2]

def roundedSurface_CF(np.ndarray[DTYPE_t,ndim=2] X, double p0, double std, double theta1, double lx, double ly, double x0, double y0):
    cdef double theta = theta1/180*pi
    
    #float[:] X_bar = X[0] - x0
    #float[:] Y_bar = X[1] - y0
    
    cdef np.ndarray[DTYPE_t] X_theta = (X[0]-x0)*cos(theta)+(X[1]-y0)*sin(theta)
    cdef np.ndarray[DTYPE_t] Y_theta = -(X[0]-x0)*sin(theta)+(X[1]-y0)*cos(theta)
    
    #cdef np.ndarray[DTYPE_t] Z = np.zeros(len(X_theta))
    
    cdef double x_theta, y_theta, r
    cdef int i
    
    for i in range(len(X_theta)):
        x_theta = X_theta[i]
        y_theta = Y_theta[i]
        
        if (fabs(x_theta) < lx/2 and fabs(y_theta) < ly/2):
            X[2][i] = p0
        elif (fabs(x_theta) > lx/2 and fabs(y_theta) > ly/2):
            r = fmin((x_theta-lx/2)**2+(y_theta-ly/2)**2,
                fmin((x_theta+lx/2)**2+(y_theta-ly/2)**2,
                fmin((x_theta-lx/2)**2+(y_theta+ly/2)**2,
                (x_theta+lx/2)**2+(y_theta+ly/2)**2)))
            r = sqrt(r)
            X[2][i] = p0*exp(-0.5*(r/std)**2)
        elif x_theta < -lx/2:
            r = -lx/2-x_theta
            X[2][i] = p0*exp(-0.5*(r/std)**2)
        elif x_theta > lx/2:
            r = x_theta-lx/2
            X[2][i] = p0*exp(-0.5*(r/std)**2)
        elif y_theta < -ly/2:
            r = -ly/2-y_theta
            X[2][i] = p0*exp(-0.5*(r/std)**2)
        elif y_theta > ly/2:
            r = y_theta - ly/2
            X[2][i] = p0*exp(-0.5*(r/std)**2)
            
    #print(X[2])
    return X[2]
"""
cdef float calc_cost(list par, np.ndarray[DTYPE_t] xdata, np.ndarray[DTYPE_t] ydata, int typee):
    cdef np.ndarray[DTYPE_t] y_func
    if typee == 1:
        y_func = roundedSurface_CF_rect(xdata,*par)
    elif typee == 2:
        y_func = roundedSurface_CF_ellips(xdata,*par)
     
    #cost = sqrt(np.sum((y_func-ydata)**2))
    return sqrt(np.sum((y_func-ydata)**2))
"""
cdef float calc_cost(list par, np.ndarray[DTYPE_t,ndim=2] xdata, np.ndarray[DTYPE_t] ydata, int typee):
    
    if typee == 1:
        y_func = roundedSurface_CF_rect(xdata,*par)
    elif typee == 2:
        y_func = roundedSurface_CF_ellips(xdata,*par)
     
    #cdef float cost = sqrt(np.sum((y_func-ydata)**2))
    return sqrt(np.sum((y_func-ydata)**2))

def Thread1_(par):
    return Thread1(*par)

def Thread2_(par):
    return Thread2(*par)

def Thread1(np.ndarray[DTYPE_t,ndim=2] xdata, np.ndarray[DTYPE_t,] ydata, list x0, list lb, list ub, list meth):
    #print("Started T1")
    #t0 = time.time()
    E_rect = list(curve_fit(roundedSurface_CF_rect, xdata, ydata, x0, bounds=(lb, ub), method = meth[0], maxfev=20, verbose=0, ftol=1e-2)[0])
    #print(time.time()-t0)
    cost_rect = calc_cost(E_rect,xdata,ydata,1)
    #print("cost_rect: "+str(cost_rect.value))
    #print("Finnished T1")
    return E_rect, cost_rect

def Thread2(np.ndarray[DTYPE_t,ndim=2] xdata,np.ndarray[DTYPE_t] ydata, list x0, list lb, list ub, list meth):
    #print("Started T2")
    #t0 = time.time()
    E_ellips = list(curve_fit(roundedSurface_CF_ellips, xdata, ydata, x0, bounds=(lb, ub), method = meth[0], maxfev=20, verbose=0, ftol=1e-2)[0])
    #print(time.time()-t0)
    cost_ellips = calc_cost(E_ellips,xdata,ydata,2)
    #print("cost_ellips: "+str(cost_ellips.value))
    #print("Finnished T2")
    return E_ellips,cost_ellips

def GaussianPressureDistribution(pressureArray, shape, spacing, t=6, n=30, x0=None, ftol=1e-3, max_it= 10, verbose_bool=False):
    """
    n: amount of points used
    """
    global x0
    n_p = min(n,shape[0]*shape[1])
    
    if verbose_bool:
        verbose = 2
    else:
        verbose = 0
    
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
    
    array_n = array[:n_p]
    
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
    
    """
    if n0 >= 5:
        x0 = [array_n[0][1],2,0,2,2,(array_n.T[2][nx[0]]+array_n.T[2][nx[1]])/2,(array_n.T[3][ny[0]]+array_n.T[3][ny[1]])/2]
    else:
        x0 = [array_n[0][1],3,0,0,0,(array_n.T[2][nx[0]]+array_n.T[2][nx[1]])/2,(array_n.T[3][ny[0]]+array_n.T[3][ny[1]])/2]
    """
    if x0 is None:
        x0 = [array_n[0][1],3,0,0,0,(array_n.T[2][nx[0]]+array_n.T[2][nx[1]])/2,(array_n.T[3][ny[0]]+array_n.T[3][ny[1]])/2]
    
     #x0 = [array_n[0][1],0.1,array_n[0][2],array_n[0][3]]
    
    #print(x0)
    
    """
    bnds = scipy.optimize.Bounds([x0[0],0.01,-90.0,0.0,0.0,min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]),min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])],
                                  [float('inf'),float('inf'),90,float('inf'),float('inf'),max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]),max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])])
    
    """
    lb = [x0[0], 1e-12, -90, 0, 0, min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    ub = [x0[0]*10, 10, 90, spacing*max(shape), spacing*max(shape), max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    
    x0 = make_feasible(x0,lb,ub)
    #time0 = time.time()
    #E = minimize(ObjFun,x0,args=(array_n,n_p),method = meth[2],bounds=bnds,options=opt_2)
    xdata = array_n.T[2:].T[:n_p].T
    xdata = np.append(xdata,np.zeros((1,xdata.shape[1])),axis=0)
    ydata = array_n.T[1][:n_p]
    
    #print(xdata)
    #print(ydata)
    meth = ['trf','dogbox']
    #pars = [xdata,ydata,x0,lb,ub,meth]
    
    """
    with ThreadPoolExecutor() as executor:
            E_rect, cost_rect = executor.submit(Thread1_,pars).result()
            E_ellips, cost_ellips = executor.submit(Thread2_,pars).result()
    
    """
    #E_rect, cost_rect = Thread1(*pars)
    #E_ellips, cost_ellips = x0, 10**10 #Thread2(*pars)
    
    
    E_rect = curve_fit(roundedSurface_CF_rect, xdata, ydata, x0, bounds=(lb, ub), method = meth[0], maxfev=20, verbose=verbose, ftol=1e-2)
    #print(E_rect)
    cost_rect = E_rect.cost
    E_ellips = curve_fit(roundedSurface_CF_ellips, xdata, ydata, x0, bounds=(lb, ub), method = meth[0], maxfev=20, verbose=verbose, ftol=1e-2)
    cost_ellips = E_ellips.cost
    
    if cost_rect < cost_ellips:
        E = E_rect
        E.x = np.append(E.x,1.0)
    else:
        E = E_ellips
        E.x = np.append(E.x,2.0)
    return array,E
    
    #E = curve_fit(roundedSurface_CF, xdata, ydata, x0, bounds=(lb, ub), method = meth[0], maxfev=1000, verbose=verbose,ftol=ftol)
    #print("time curve fitting: "+str(time.time()-time0))
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
    #return array,E
