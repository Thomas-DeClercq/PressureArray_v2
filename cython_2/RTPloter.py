#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:50:25 2022

@author: pc-robotiq
"""
import rospy
from std_msgs.msg import Float64MultiArray

import numpy as np
import math
from math import pi,sin,cos,exp,sqrt

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib import cm

params = np.zeros((7,))

def readTopic(data):
    global params
    for i,_ in enumerate(params):
        params[i] = data.data[i]
    #print(params)
    return

def gaussian_r(r, p0, std):
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

def animate(i,X,Y):
    global params
    global cont
    
    #freq = 10
    #rate = rospy.Rate(freq)
    rospy.Subscriber('/estimatedParameters',Float64MultiArray,readTopic,queue_size=1)
    #print(params)
    
    Z = roundedSurface_array(X,Y,*params)
    for c in cont.collections:
        c.remove()
    cont = ax.contourf(X,Y,Z,cmap = cm.coolwarm,vmin=0,vmax=5e4)
    #rate.sleep()
    return cont,


if __name__ == '__main__':
    rospy.init_node('RTPloter')
    freq = 10
    rate = rospy.Rate(freq)
    
    shape = [3,3]
    spacing = 4 # mm
    
    fig = plt.figure(figsize=[shape[0]*2,shape[1]*2])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    
    num_cir = int(shape[0]*shape[1])
    
    X = np.linspace(0, (shape[0]+1)*spacing, 20)
    Y = np.linspace(0, (shape[1]+1)*spacing, 20)
    X, Y = np.meshgrid(X, Y)
    Z = roundedSurface_array(X,Y,*params)
    cont = ax.contourf(X,Y,Z,cmap = cm.coolwarm,vmin=0,vmax=5e4)
    #ax.set_clim(0,1e5)
    #cbar = fig.colorbar(cont)
    
    for i in range(num_cir):
        x = spacing+spacing*(i % shape[0])
        y = spacing+spacing*(math.floor(i/shape[0]) % shape[1])
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
    
    
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  fargs=(X,Y),
                                  interval=1/freq*1000)
                                  #blit=True)
    """
    for _ in range(10):
        rospy.Subscriber('/estimatedParameters',Float64MultiArray,readTopic,queue_size=1)
    """
    plt.show()