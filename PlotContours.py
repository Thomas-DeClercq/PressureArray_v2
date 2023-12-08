#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:20:31 2021

@author: pc-robotiq
"""
from math import exp,pi,cos,sin,sqrt,atan
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import numpy as np

def gaussian_r(r,p0,std):
    return p0*exp(-0.5*(r/std)**2)

def gaussian_2d(x,y,p0,std_x,std_y,rho,x0,y0):
    return p0*np.exp(-0.5/(1-rho**2)*(((x-x0)/std_x)**2-2*rho*((x-x0)/std_x)*((y-y0)/std_y)+((y-y0)/std_y)**2))

def roundedSurface_2(X,Y,p0,std_x,std_y,theta,lx,ly,x0,y0):
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
                dx = min(abs(x_theta-lx/2),abs(x_theta+lx/2))
                dy = min(abs(y_theta-ly/2),abs(y_theta+ly/2))
                angle = atan(dy/dx)/pi*180
                std = angle/90*std_y + (90-angle)/90*std_x
                #std = std_y**(angle/90)*std_x**(1-angle/90)
                Z[i][j] = gaussian_r(sqrt(r),p0,std)
            elif x_theta < -lx/2:
                Z[i][j] = gaussian_r(-lx/2-x_theta,p0,std_x)
            elif x_theta > lx/2:
                Z[i][j] = gaussian_r(x_theta-lx/2,p0,std_x)
            elif y_theta < -ly/2:
                Z[i][j] = gaussian_r(-ly/2-y_theta,p0,std_y)
            elif y_theta > ly/2:
                Z[i][j] = gaussian_r(y_theta-ly/2,p0,std_y)
    return Z

def roundedSurface(X,Y,p0,std,theta,lx,ly,x0,y0):
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

def rotatedGuassian(X,Y,p0,std_x,std_y,theta,x0,y0):
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


shape = [5,9]
spacing = 4.5

fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
ax = plt.subplot()
ax.set_xlim([0,spacing*(shape[0]+1)])
ax.set_ylim([0,spacing*(shape[1]+1)])

X = np.arange(0, (shape[0]+1)*spacing, 0.1)
Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
X, Y = np.meshgrid(X, Y)

#Z = gaussian_2d(X,Y,63000,8,4,-0.9,8,29) 
Z = roundedSurface(X,Y,10,2,45,5,10,13.5,22.5)
#Z = roundedSurface_2(X,Y,1,5,2,0,10,2,13.5,22.5)
#Z = rotatedGuassian(X,Y,1,5,2,-35,8,29)

cont = plt.contourf(X,Y,Z,cmap = cm.coolwarm,extend='both',vmin=0,vmax=10)

num_cir = shape[0]*shape[1]
for i in range(num_cir):
        x = spacing+spacing*(i % shape[0])
        y = spacing+spacing*(i % shape[1])
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))

plt.tick_params(left=False,right=False,labelleft=False,labelbottom=False,bottom=False)
cbar = plt.colorbar(cont,extendrect=False,ticks=[0,2,4,6,8,10])
cbar.ax.tick_params(labelsize=30)
cbar.ax.set_ylabel('Pressure [kPa]', rotation=90, fontsize=30)
cbar.ax.set_ylim(0,10)
plt.savefig(f'./cbar.png', format='png',bbox_inches='tight')

plt.show()