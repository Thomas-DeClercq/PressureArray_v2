#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:20:31 2021

@author: pc-robotiq
"""
from math import exp,pi,cos,sin
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
import numpy as np

def gaussian(x,y,p0,std,x0,y0):
    return p0*exp(-0.5*(((x-x0)**2+(y-y0)**2)/(std**2)))

def gaussian_2d(x,y,p0,std_x,std_y,rho,x0,y0):
    return p0*np.exp(-0.5/(1-rho**2)*(((x-x0)/std_x)**2-2*rho*((x-x0)/std_x)*((y-y0)/std_y)+((y-y0)/std_y)**2))

def roundedSurface(X,Y,p0,std,theta,lx,ly,x0,y0):
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i][j]
            y = Y[i][j]
            theta = theta/180*pi
            
            x_bar = x - x0
            y_bar = y - y0
            
            x_theta = x_bar*cos(theta)-y_bar*sin(theta)
            y_theta = x_bar*sin(theta)+y_bar*cos(theta)
            
            if (abs(x_theta) < lx/2 and abs(y_theta) < ly/2):
                Z[i][j] = p0
            
            
    return Z

shape = [5,9]
spacing = 4.5


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlim([0, shape[0]*spacing])
ax.set_ylim([0, shape[1]*spacing])
ax.set_aspect('equal')

# Make data.
X = np.arange(0, shape[0]*spacing, 0.1)
Y = np.arange(0, shape[1]*spacing, 0.1)
X, Y = np.meshgrid(X, Y)

#Z = gaussian_2d(X,Y,1,5,5,0.99,12,20)
Z = roundedSurface(X,Y,1,1,0,5,10,13.5,22.5)

surf = ax.plot_surface(X, Y, Z,  cmap=cm.coolwarm, linewidth=0, antialiased=False)
