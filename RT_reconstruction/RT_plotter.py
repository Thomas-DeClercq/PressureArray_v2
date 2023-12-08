import rospy
from std_msgs.msg import Float64MultiArray

import numpy as np
import math
from math import pi,sin,cos,exp,sqrt

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib import cm

import sys
sys.path.insert(1,'/home/thomas/pythonScripts/PressureArray_v2/PR_cython')
from PressureReconstruction_141123 import Optimization, calc_Z

params = np.zeros((1,8))
params[0][1] = 1

def readTopic(data):
    global params
    rows = int(len(data.data)/8)
    params = np.zeros((rows,8))


    for i in range(rows):
        for j in range(8):
            params[i][j] = data.data[i*8+j]
    #print(params)
    return

def animate(i,X,Y):
    global params
    global cont
    
    #freq = 10
    #rate = rospy.Rate(freq)
    #print(params)

    Z = np.zeros(X.flatten().shape)
    for idx in range(params.shape[0]):
        try:
            Z_add = calc_Z(X.flatten(),Y.flatten(),*params[idx])
        except Exception as e:
            Z_add = np.zeros(X.flatten().shape)
            print(f"{e} with parameters: {params[idx]}")
        Z = Z + Z_add
    Z = np.reshape(Z, X.shape)

    for c in cont.collections:
        c.remove()
    """
    for i in range(num_cir):
        x = spacing+spacing*(i % shape[0])
        y = spacing+spacing*(math.floor(i/shape[0]) % shape[1])
        ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
    """
    cont = ax.contourf(X,Y,Z,cmap = cm.coolwarm,vmin=0) #,vmax=5e4)
    #rate.sleep()
    return cont,


if __name__ == '__main__':
    rospy.init_node('RTPloter')
    freq = 10
    rate = rospy.Rate(freq)
    rospy.Subscriber('/estimatedParameters',Float64MultiArray,readTopic,queue_size=1)
    
    shape = [4,8]
    spacing = 4.5 # mm
    
    fig = plt.figure(figsize=[shape[0]*2,shape[1]*2])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    ax.set_aspect('equal')
    
    num_cir = int(shape[0]*shape[1])
    
    X = np.linspace(0, (shape[0]+1)*spacing, 20)
    Y = np.linspace(0, (shape[1]+1)*spacing, 20)
    X, Y = np.meshgrid(X, Y)
    Z =  calc_Z(X.flatten(),Y.flatten(),*params[0])
    Z = np.reshape(Z, X.shape)
    cont = ax.contourf(X,Y,Z,cmap = cm.coolwarm,vmin=0)#,vmax=5e4)
    #ax.set_clim(0,1e5)
    #cbar = fig.colorbar(cont)
    
    """
    for i in range(num_cir):
        x = spacing+spacing*(i % shape[0])
        y = spacing+spacing*(math.floor(i/shape[0]) % shape[1])
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
        ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
    """
    
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  fargs=(X,Y),
                                  interval=1/freq*1000)
                                  #blit=True)

    plt.show()
