#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:44:52 2022

@author: pc-robotiq
"""
import random, math
import numpy as np
import pandas as pd
from math import sin, cos, pi

from matplotlib import pyplot as plt
from matplotlib import cm, animation
import matplotlib.patches as patches

from PressureReconstruction import calc_Z

def plot_params(shape,spacing,X,Y,params):
    Z_n = np.zeros(X.shape)
    for i in range(params.shape[0]):
        Z_1 = calc_Z(X.flatten(),Y.flatten(),*params_i)
        Z_1 = np.reshape(Z_1,X.shape)
        Z_n = Z_n + Z_1
        
    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    """
    X = np.arange(0, (shape[0]+1)*spacing, 0.05)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.05)
    X, Y = np.meshgrid(X, Y)
    """
    plt.contourf(X,Y,Z_n,cmap = cm.coolwarm,extend='both')
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

def RandomParams(shape,spacing):
    """
    lb = [x0[0], 0, -90, 0, 0, min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    ub = [x0[0]*10, 10, 90, spacing*max(shape), spacing*max(shape), max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    allInOne(X,Y,p0,std,lx,ly,S_x,S_y,S,r_curve,F,theta,x0,y0):
    """
    p0 = random.uniform(10,100)
    std = random.uniform(1,10)
    lx = random.uniform(spacing,spacing*max(shape)/2)
    ly = random.uniform(0,lx)
    #S_x = random.uniform(1,10)
    #S_y = random.uniform(1,10)
    #S = random.uniform(0,1)
    r_curve = random.uniform(0,3)
    #F = random.uniform(1,1)
    theta = random.uniform(-90,90)
    x0 = random.uniform(spacing,spacing*(shape[0]-1))
    y0 = random.uniform(spacing,spacing*(shape[1]-1))
    return [p0,std,lx,ly,r_curve,theta,x0,y0]#[p0,std,lx,ly,S_x,S_y,S,r_curve,F,theta,x0,y0]

def static(params):
    for t in range(1,params.shape[0]):
        for it in range(params.shape[1]):
            for par in range(params.shape[2]):
                factor = random.uniform(0.9,1.1)
                params[t][it][par] = factor*params[0][it][par]
    all_x0 = np.take(params,[6],axis=2).flatten()
    all_y0 = np.take(params,[7],axis=2).flatten()
    if np.max(all_x0) < 22.5 and np.min(all_x0) > 4.5 and np.max(all_y0) < 40.5 and np.min(all_y0) > 4.5:
        return True, params
    else:
        return False, params

def transational_slip(params,dist,alpha=0):
    dist_t = dist/params.shape[0]
    for t in range(1,params.shape[0]):
        for it in range(params.shape[1]):
            for par in range(params.shape[2]-2):
                factor = random.uniform(0.95,1.05)
                params[t][it][par] = factor*params[0][it][par]
            params[t][it][6] = params[0][it][6] + t*dist_t*cos(alpha/180*pi)
            params[t][it][7] = params[0][it][7] + t*dist_t*sin(alpha/180*pi)
    all_x0 = np.take(params,[6],axis=2).flatten()
    all_y0 = np.take(params,[7],axis=2).flatten()
    if np.max(all_x0) < 22.5 and np.min(all_x0) > 4.5 and np.max(all_y0) < 40.5 and np.min(all_y0) > 4.5:
        return True, params
    else:
        return False, params
   
def rotational_slip(params,point,angle):
    angle = angle/180*pi
    angle_t = angle/params.shape[0]
    angle_t_deg = angle_t*180/pi
    x_r = point[0]
    y_r = point[1]
    for t in range(1,params.shape[0]):
        for it in range(params.shape[1]):
            for par in range(params.shape[2]-3):
                factor = random.uniform(0.95,1.05)
                params[t][it][par] = factor*params[0][it][par]
            params[t][it][5] = params[0][it][5] + t*angle_t_deg
            params[t][it][6] = cos(t*angle_t)*params[0][it][6] - sin(t*angle_t)*params[0][it][7] + x_r - cos(t*angle_t)*x_r + sin(t*angle_t)*y_r
            params[t][it][7] = sin(t*angle_t)*params[0][it][6] + cos(t*angle_t)*params[0][it][7] + y_r - sin(t*angle_t)*x_r - cos(t*angle_t)*y_r
        all_x0 = np.take(params,[6],axis=2).flatten()
    all_y0 = np.take(params,[7],axis=2).flatten()
    if np.max(all_x0) < 22.5 and np.min(all_x0) > 4.5 and np.max(all_y0) < 40.5 and np.min(all_y0) > 4.5:
        return True, params
    else:
        return False, params
    
def pure_rotational_slip(params,angle):
    angle = angle/180*pi
    angle_t = angle/params.shape[0]
    angle_t_deg = angle_t*180/pi
    x_r = np.average(np.multiply(params[0].T[6],params[0].T[0]))/np.average(params[0].T[0])
    y_r = np.average(np.multiply(params[0].T[7],params[0].T[0]))/np.average(params[0].T[0]) 
    for t in range(1,params.shape[0]):
        for it in range(params.shape[1]):
            for par in range(params.shape[2]-3):
                factor = random.uniform(0.95,1.05)
                params[t][it][par] = factor*params[0][it][par]
            params[t][it][5] = params[0][it][5] + t*angle_t_deg
            params[t][it][6] = cos(t*angle_t)*params[0][it][6] - sin(t*angle_t)*params[0][it][7] + x_r - cos(t*angle_t)*x_r + sin(t*angle_t)*y_r
            params[t][it][7] = sin(t*angle_t)*params[0][it][6] + cos(t*angle_t)*params[0][it][7] + y_r - sin(t*angle_t)*x_r - cos(t*angle_t)*y_r
    all_x0 = np.take(params,[6],axis=2).flatten()
    all_y0 = np.take(params,[7],axis=2).flatten()
    if np.max(all_x0) < 22.5 and np.min(all_x0) > 4.5 and np.max(all_y0) < 40.5 and np.min(all_y0) > 4.5:
        return True, params
    else:
        return False, params

def rotational_and_transational_slip(params,angle,dist,alpha=0):
    dist_t = dist/params.shape[0]
    angle = angle/180*pi
    angle_t = angle/params.shape[0]
    angle_t_deg = angle_t*180/pi
    x_r = np.average(np.multiply(params[0].T[6],params[0].T[0]))/np.average(params[0].T[0])
    y_r = np.average(np.multiply(params[0].T[7],params[0].T[0]))/np.average(params[0].T[0]) 
    for t in range(1,params.shape[0]):
        for it in range(params.shape[1]):
            for par in range(params.shape[2]-3):
                factor = random.uniform(0.95,1.05)
                params[t][it][par] = factor*params[0][it][par]
            params[t][it][5] = params[0][it][5] + t*angle_t_deg
            params[t][it][6] = cos(t*angle_t)*params[0][it][6] - sin(t*angle_t)*params[0][it][7] + x_r - cos(t*angle_t)*x_r + sin(t*angle_t)*y_r + t*dist_t*cos(alpha/180*pi)
            params[t][it][7] = sin(t*angle_t)*params[0][it][6] + cos(t*angle_t)*params[0][it][7] + y_r - sin(t*angle_t)*x_r - cos(t*angle_t)*y_r + t*dist_t*sin(alpha/180*pi)
    all_x0 = np.take(params,[6],axis=2).flatten()
    all_y0 = np.take(params,[7],axis=2).flatten()
    if np.max(all_x0) < 22.5 and np.min(all_x0) > 4.5 and np.max(all_y0) < 40.5 and np.min(all_y0) > 4.5:
        return True, params
    else:
        return False, params

def animate(i,X,Y,all_params):
    global cont
    
    Z_n = np.zeros(X.shape)
    for it in range(all_params.shape[1]):
        Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params[i][it])
        Z_1 = np.reshape(Z_1,X.shape)
        Z_n = Z_n + Z_1
    
    for c in cont.collections:
        c.remove()
    cont = ax.contourf(X,Y,Z_n,cmap=cm.coolwarm,vmin=0)
    TXT_t.set_text(str("t = ")+str(i))
    return cont,

def writeableData(params):
    timesteps = params.shape[0]
    iters = params.shape[1]
    amount_par = params.shape[2]
    
    df = np.zeros((timesteps*iters,amount_par+2))
    for i in range(timesteps*iters):
        t = math.floor(i/iters)
        it = i%iters
        df[i][0] = t
        df[i][1] = it
        for j in range(amount_par):
            df[i][2+j] = params[t][it][j]
       
    columns = ['Timestep','iteration','p0','std','lx','ly','r_curve','theta','x0','y0']
    df = pd.DataFrame(df,columns=columns)
    return df

if __name__ == "__main__":
    
    shape = [5,9]
    spacing = 4.5
    plot_animation = True
    
    
    cases = ['static','trans','rot','trans_rot']
    cases = ['rot']
    #case = "trans_rot" #static,trans,rot,trans_rot
    
    n = 50        
    params_names = ['p0','std','lx','ly','r_curve','theta','x0','y0']
  
    timesteps = 2
    respect_boundaries=False
    
    for case in cases:
        for it in [3]:
            params_0 = np.zeros((it,8))
            all_params = np.zeros((timesteps,it,8))
            j = 0            
            while (j < 1):
                for i in range(it):
                    params_i = RandomParams(shape,spacing)
                    params_0[i] = params_i
                    
                all_params[0] = params_0
                
                if case == 'static':
                    respect_boundaries, all_params = static(all_params)
                    case_name = case+"_"+str(it)+"_"+str(j)
                elif case == 'trans':
                    dist = random.uniform(-20,20)
                    alpha = random.uniform(-180,180)
                    respect_boundaries, all_params = transational_slip(all_params,dist,alpha)
                    case_name = case+"_"+str(dist)+"_"+str(alpha)+"_"+str(it)
                elif case == 'rot':
                    angle = random.uniform(90,180)
                    respect_boundaries, all_params = pure_rotational_slip(all_params, angle)
                    case_name = case+"_"+str(angle)+"_"+str(it)
                elif case == 'trans_rot':
                    angle = random.uniform(120,180)
                    dist = random.uniform(10,20)
                    alpha = random.uniform(-180,180)
                    respect_boundaries, all_params = rotational_and_transational_slip(all_params,angle,dist,alpha)
                    case_name = case+"_"+str(dist)+"_"+str(alpha)+"_"+str(angle)+"_"+str(it)
                else:
                    print("Unknown case selected")
                
                # generate animiation
                if plot_animation and respect_boundaries:
                    X = np.linspace(0, (shape[0]+1)*spacing, n)
                    Y = np.linspace(0, (shape[1]+1)*spacing, n)
                    X, Y = np.meshgrid(X, Y)
                    
                    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
                    ax = plt.subplot()
                    ax.set_xlim([0,spacing*(shape[0]+1)])
                    ax.set_ylim([0,spacing*(shape[1]+1)])
                    
                    num_cir = int(shape[0]*shape[1])
                    for i in range(num_cir):
                        x = spacing+spacing*(i % shape[0])
                        y = spacing+spacing*(i % shape[1])
                        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
                        #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
                        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
                        #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
                        ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
                    
                    Z_n = np.zeros(X.shape)
                    for itt in range(all_params.shape[1]):
                        Z_1 = calc_Z(X.flatten(),Y.flatten(),*all_params[0][itt])
                        Z_1 = np.reshape(Z_1,X.shape)
                        Z_n = Z_n + Z_1
                    
                    cont = ax.contourf(X,Y,Z_n,cmap=cm.coolwarm,vmin=0)
                    
                    TXT_t = plt.text(13.5,-4,str("t = 0"),size=15,horizontalalignment='center')
                    
                    ani = animation.FuncAnimation(fig,
                                                  animate,
                                                  fargs=(X,Y,all_params),
                                                  interval=1000,
                                                  frames = timesteps,
                                                  repeat=True,
                                                  repeat_delay=0)
                
                if respect_boundaries:
                    print('ok')
                    j += 1
                    df = writeableData(all_params)
                    #df.to_csv("./PressureDataInTime_DEMOS/"+case_name+".csv",index=False)
                else:
                    print('not ok')
                
    
    
        
        
    