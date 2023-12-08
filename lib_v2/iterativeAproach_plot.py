#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:02:36 2022

@author: pc-robotiq
Script was used for creating the parameterised curved pressure distributions
"""
import os
import numpy as np
import pandas as pd
import scipy
import math
from math import sqrt,pi,exp,sin,cos,log
from scipy.optimize import Bounds, minimize, curve_fit
#from customScipy import curve_fit
from matplotlib import cm
import matplotlib.patches as patches
import time
import random

#import nlopt

#from ROS_LocAndForceEstimation import HerzianContactLoc
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
    
def plot_Z(shape,spacing,X,Y,Z,Z_max=np.array([])):
    if not Z_max.any():
        Z_max = Z
    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    """
    X = np.arange(0, (shape[0]+1)*spacing, 0.05)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.05)
    X, Y = np.meshgrid(X, Y)
    """
    plt.contourf(X,Y,Z,cmap = cm.coolwarm,extend='both',vmin=0,vmax=np.max(Z_max))
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
    
def gaussian_r(r,p0,std):
    return p0*exp(-0.5*(r/std)**2)

def allInOne(X,Y,p0,std,lx,ly,r_curve,theta,x0,y0):
    Z = np.zeros(X.shape)
    Z_new = Z.copy()
    
    X_bar = X - 13.5
    Y_bar = Y - 22.5
    
    theta = theta/180*pi
        
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X_bar[i][j]# - 13.5
            y = Y_bar[i][j]# - 22.5
            
            if (abs(x) < lx/2 and abs(y) < ly/2):
                Z[i][j] = p0
            elif (abs(x) > lx/2 and abs(y) > ly/2):
                r = min((x-lx/2)**2+(y-ly/2)**2,
                        (x+lx/2)**2+(y-ly/2)**2,
                        (x-lx/2)**2+(y+ly/2)**2,
                        (x+lx/2)**2+(y+ly/2)**2)
                Z[i][j] = gaussian_r(sqrt(r),p0,std)
            elif x < -lx/2:
                Z[i][j] = gaussian_r(-lx/2-x,p0,std)
            elif x > lx/2:
                Z[i][j] = gaussian_r(x-lx/2,p0,std)
            elif y < -ly/2:
                Z[i][j] = gaussian_r(-ly/2-y,p0,std)
            elif y > ly/2:
                Z[i][j] = gaussian_r(y-ly/2,p0,std)
            else:
                Z[i][j] = p0
            
    if r_curve == 0:
        X_new = X_bar.copy()
        Y_new = Y_bar.copy()
    elif r_curve > 0:
        r = Y_bar + r_curve
        alpha = np.divide(X_bar,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    else:
        r = -Y_bar - r_curve
        alpha = np.divide(X_bar,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
        
    X_f = X_new*cos(theta)+Y_new*sin(theta) + x0
    Y_f = -X_new*sin(theta)+Y_new*cos(theta) + y0
                
    return X_f,Y_f,Z

def allInOne_alt(X,Y,p0,std,lx,ly,r_curve,theta,x0,y0):
    Z = np.zeros(X.shape)
    
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    X_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    Y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
    
    """
    if r_curve == 0:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
    elif r_curve > 0:
        r = Y_theta + r_curve
        alpha = np.divide(X_theta,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    else:
        r = -Y_theta- r_curve
        alpha = np.divide(X_theta,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    """
    if r_curve < -0.1:
        r_curve = 10/r_curve
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r + r_curve)
    elif r_curve > 0.1:
        r_curve = 10/r_curve
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,-(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r - r_curve)
    else:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
        
    #X_new2 = np.reshape(X_new,(20,20))
    #Y_new2 = np.reshape(Y_new,(20,20))
      
    for i,(x,y) in enumerate(zip(X_new,Y_new)):
        if (abs(x) < lx/2 and abs(y) < ly/2):
            Z[i] = p0
        elif (abs(x) > lx/2 and abs(y) > ly/2):
            r = min((x-lx/2)**2+(y-ly/2)**2,
                    (x+lx/2)**2+(y-ly/2)**2,
                    (x-lx/2)**2+(y+ly/2)**2,
                    (x+lx/2)**2+(y+ly/2)**2)
            Z[i] = gaussian_r(sqrt(r),p0,std)
        elif x < -lx/2:
            Z[i] = gaussian_r(-lx/2-x,p0,std)
        elif x > lx/2:
            Z[i] = gaussian_r(x-lx/2,p0,std)
        elif y < -ly/2:
            Z[i] = gaussian_r(-ly/2-y,p0,std)
        elif y > ly/2:
            Z[i] = gaussian_r(y-ly/2,p0,std)
        else:
            Z[i] = p0
    
    """
    if r_curve == 0:
        X_new = X_bar.copy()
        Y_new = Y_bar.copy()
    elif r_curve > 0:
        r = Y_bar + r_curve
        alpha = np.divide(X_bar,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    else:
        r = -Y_bar - r_curve
        alpha = np.divide(X_bar,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
        
    X_f = X_new*cos(theta)+Y_new*sin(theta) + x0
    Y_f = -X_new*sin(theta)+Y_new*cos(theta) + y0
    """       
    return Z

def allInOne_alt3(X,Y,p0,std,lx,ly,r_curve,theta,x0,y0):
    Z = np.zeros(X.shape)
    
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    X_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    Y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
    
    """
    if r_curve == 0:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
    elif r_curve > 0:
        r = Y_theta + r_curve
        alpha = np.divide(X_theta,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    else:
        r = -Y_theta- r_curve
        alpha = np.divide(X_theta,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    """
    if r_curve < -0.1:
        r_curve = -10/(10**(-r_curve-1))
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r + r_curve)
    elif r_curve > 0.1:
        r_curve = 10/(10**(r_curve-1))
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,-(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r - r_curve)
    else:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
        
    #X_new2 = np.reshape(X_new,(20,20))
    #Y_new2 = np.reshape(Y_new,(20,20))
      
    for i,(x,y) in enumerate(zip(X_new,Y_new)):
        if (abs(x) < lx/2 and abs(y) < ly/2):
            Z[i] = p0
        elif (abs(x) > lx/2 and abs(y) > ly/2):
            r = min((x-lx/2)**2+(y-ly/2)**2,
                    (x+lx/2)**2+(y-ly/2)**2,
                    (x-lx/2)**2+(y+ly/2)**2,
                    (x+lx/2)**2+(y+ly/2)**2)
            Z[i] = gaussian_r(sqrt(r),p0,std)
        elif x < -lx/2:
            Z[i] = gaussian_r(-lx/2-x,p0,std)
        elif x > lx/2:
            Z[i] = gaussian_r(x-lx/2,p0,std)
        elif y < -ly/2:
            Z[i] = gaussian_r(-ly/2-y,p0,std)
        elif y > ly/2:
            Z[i] = gaussian_r(y-ly/2,p0,std)
        else:
            Z[i] = p0
    
    """
    if r_curve == 0:
        X_new = X_bar.copy()
        Y_new = Y_bar.copy()
    elif r_curve > 0:
        r = Y_bar + r_curve
        alpha = np.divide(X_bar,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
    else:
        r = -Y_bar - r_curve
        alpha = np.divide(X_bar,r_curve)
        X_new = np.multiply(r,np.sin(alpha))
        Y_new = np.multiply(r,np.cos(alpha)) - r_curve
        
    X_f = X_new*cos(theta)+Y_new*sin(theta) + x0
    Y_f = -X_new*sin(theta)+Y_new*cos(theta) + y0
    """       
    return Z


def allInOne_alt2(X,Y,p0,std,lx,ly,r_curve,theta,x0,y0):
    Z = np.zeros(X.shape)
    theta = theta/180*pi
    
    theta = theta/180*pi
    
    X_bar = X - x0
    Y_bar = Y - y0
    
    X_theta = X_bar*cos(theta)+Y_bar*sin(theta)
    Y_theta = -X_bar*sin(theta)+Y_bar*cos(theta)
    
    if r_curve < -0.1:
        r_curve = 10/r_curve
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r + r_curve)
    elif r_curve > 0.1:
        r_curve = 10/r_curve
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,-(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r - r_curve)
    else:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
    
    for i in range(X_new.shape[0]):
        for j in range(X_new.shape[1]):
            """
            x = X[i][j]
            y = Y[i][j]
            x_bar = x - x0
            y_bar = y - y0
            """
            x_theta = X_new[i][j]
            y_theta = Y_new[i][j]
            
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
            else:
                Z[i][j] = p0
    return Z

def getValue(X,Y,Z,xi,yi):
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    point = [(xi,yi)]
    zi = scipy.interpolate.griddata((X,Y),Z,point)
    zi = np.nan_to_num(zi)
    return zi

def RandomParams(shape,spacing):
    """
    lb = [x0[0], 0, -90, 0, 0, min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    ub = [x0[0]*10, 10, 90, spacing*max(shape), spacing*max(shape), max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    allInOne(X,Y,p0,std,lx,ly,S_x,S_y,S,r_curve,F,theta,x0,y0):
    """
    p0 = random.uniform(10000,100000)
    std = random.uniform(0,10)
    lx = random.uniform(0,spacing*max(shape)/3)
    ly = random.uniform(0,lx)
    r_curve = random.uniform(-3,3)
    theta = random.uniform(-90,90)
    x0 = random.uniform(0,spacing*(shape[0]+1))
    y0 = random.uniform(0,spacing*(shape[1]+1))
    return [p0,std,lx,ly,r_curve,theta,x0,y0]

"""
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
"""

if __name__ == '__main__':
    
    shape = [5,9]
    spacing = 4.5
    
    X = np.linspace(0, (shape[0]+1)*spacing, 100)
    Y = np.linspace(0, (shape[1]+1)*spacing, 100)
    X, Y = np.meshgrid(X, Y)
    
    Z_all = np.zeros((100,100))
    list_time = []
        
    for ind in range(3):#len(list_files)):
        print("------------------------------------------------")
        print(ind+1)
        t0 = time.time()
    
        params_names = ['p0','std','lx','ly','r_curve','theta','x0','y0']
        params = RandomParams(shape,spacing)
        
        #params = [10,2,0,10,0.5,13.5,22.5]
        #params = [10,0.5,15,0,1.5,0,13.5,22.5]
        #params = [10,2,80,2,20,0,13.5,22.5]
        print(dict(zip(params_names,params)))
        
        X = np.linspace(0, (shape[0]+1)*spacing, 100)
        Y = np.linspace(0, (shape[1]+1)*spacing, 100)
        X, Y = np.meshgrid(X, Y)
        
        """
        #Z = roundedSurface_array(X,Y,params[0],params[1],params[2],params[3],params[4],params[5],params[6])
        X_0,Y_0,Z_0 = Surface(X,Y,10,1,15,2)
        plot_Z(shape,spacing,X_0,Y_0,Z_0)
        
        X_1,Y_1,Z_1 = Scaling(X_0,Y_0,Z_0,1,10,0.1)
        plot_Z(shape,spacing,X_1,Y_1,Z_1)
        
        X_2,Y_2,Z_2 = Curve(X_1,Y_1,Z_1,0,0)
        plot_Z(shape,spacing,X_2,Y_2,Z_2,Z_0)
                
        X_3,Y_3,Z_3 = RotateAndTranslate(X_2,Y_2,Z_2,0,13.5,22.5)
        plot_Z(shape,spacing,X_3,Y_3,Z_3,Z_3)
        #plot pressure distribitution
        #plot_roundedSurface(shape,spacing,params)
        """
        #X_f,Y_f,Z_f = allInOne(X,Y,10,1,15,2,2,1,0.1,5,2,45,13.5,22.5)
        #Z = roundedSurface_array(X,Y,10,1,45,15,2,10,30)
        #plot_Z(shape,spacing,X,Y,Z)
        
        #X_f,Y_f,Z_f = allInOne(X,Y,*params)
        #plot_Z(shape,spacing,X_f,Y_f,Z_f,Z_f)
        
        Z_alt = allInOne_alt3(X.flatten(),Y.flatten(),*params)
        Z_alt = np.reshape(Z_alt,X.shape)
        #Z_alt = allInOne_alt2(X,Y,*params)
        Z_all = Z_all + Z_alt
        
        plot_Z(shape,spacing,X,Y,Z_alt)
        #results = np.zeros((4,1))
    
        #array, E = GaussianPressureDistribution(pres.copy(),shape,spacing,n=30)
        dur = time.time()-t0
        
        #print(E.x)
        #print('duration: '+str(dur))
        list_time.append(dur)
    
    plot_Z(shape,spacing,X,Y,Z_all)
    
    print(":::::::::::::::::::::::::::::::::::")
    #print(sum(list_errors)/len(list_errors))
    list_time = np.array(list_time)
    print("mean duration: "+str(list_time.mean()))
