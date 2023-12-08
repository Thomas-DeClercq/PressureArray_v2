#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:37:08 2021

@author: pc-robotiq
"""

import time
import numpy as np
import math

import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

#freq = 30
coord = np.zeros((2,))
F = 0

def readCoord(value):
    global coord
    coord[0] = value.data[0]
    coord[1] = value.data[1]
    return

def readForce(value):
    global F
    F = value.data
    return

def animate(i):
    global coord
    global F
    global PP
    global TXT
    
    
    #print(time.time()-t_start)
    #t_start = time.time()
    
    rospy.Subscriber('/estimatedLocation',Float64MultiArray,readCoord,queue_size=1)
    rospy.Subscriber('/estimatedForce',Float64,readForce,queue_size=1)
    
    if F < 0:
        PP.center = (coord[0],coord[1])
    else:
        PP.center = (-10,-10)
    TXT.set_text(f"{F:.2f} N")

    return PP,TXT,
         

        
if __name__ == '__main__':
    rospy.init_node('RealTimePlotter')
    freq = 20
    rate = rospy.Rate(freq)
    
    calibrated = False
    shape = [4,8]
    spacing = 4.5 # mm

    fig = plt.figure(figsize=[shape[0]*2,shape[1]*2])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    ax.set_aspect('equal')
    
    num_cir = int(shape[0]*shape[1])
    
    PP = plt.Circle((-10,-10),0.5,fill=True,edgecolor='yellow',facecolor='yellow')
    ax.add_patch(PP)
    TXT = plt.text(11.25,1,f"0 N", ha = 'center', size='large')
    
    
    for idx in range(num_cir):
        x = spacing + spacing*(idx % shape[0])
        y = spacing*(shape[1]) - spacing*math.floor(idx/shape[0])
        ax.add_patch(patches.Circle((x,y),0.5,fill=True,edgecolor='red',facecolor='red'))

        
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  fargs=(),
                                  interval=1/freq*1000,
                                  blit=True)
        
    plt.show()
