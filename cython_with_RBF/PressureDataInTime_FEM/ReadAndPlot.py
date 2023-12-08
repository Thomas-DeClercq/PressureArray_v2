#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:25:58 2021

@author: pc-robotiq
"""
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import numpy as np
import scipy.interpolate
import os

def conv(x):
    return x.replace(',', '.').encode()

name_dir = "./S-shape/"
list_files = os.listdir(name_dir)
#for file in list_files:
i=0
while (i < len(list_files)):
    if '.' in list_files[i]:
        list_files.remove(list_files[i])
    else:
        i += 1
#list_files.remove('.~lock.0_0.csv#')
#name_file = "/home/pc-robotiq/measurements_paper/FEM/csvs/"+name_file+".csv"
for name_file in list_files:
    #name_file = list_files[0]
    print(name_file)
    
    raw_pressure = np.genfromtxt((conv(x) for x in open(name_dir+str(name_file)+"/Pressure.txt")),delimiter='\t',skip_header=1)
    raw_x_coord = np.genfromtxt((conv(x) for x in open(name_dir+str(name_file)+"/x_coord.txt")),delimiter='\t',skip_header=1)
    raw_y_coord = np.genfromtxt((conv(x) for x in open(name_dir+str(name_file)+"/y_coord.txt")),delimiter='\t',skip_header=1)
    
    all_data = np.zeros([len(raw_pressure),4])
    for i in range(len(raw_pressure)):
        node = int(raw_pressure[i][0])
        all_data[i][0] = node
        all_data[i][3] = raw_pressure[i][1]*-1*10**6
        
        all_data[i][1] = raw_x_coord[node-1][1]+13.5
        all_data[i][2] = raw_y_coord[node-1][1]+22.5
    
            
    shape = [5,9]
    spacing = 4.5
    
    fig = plt.figure(figsize=[(shape[0]+1),(shape[1]+1)])
    ax = plt.subplot()
    ax.set_xlim([0,spacing*(shape[0]+1)])
    ax.set_ylim([0,spacing*(shape[1]+1)])
    
    x = all_data.T[1]
    y = all_data.T[2]
    z = all_data.T[3]
    
    X = np.arange(0, (shape[0]+1)*spacing, 0.1)
    Y = np.arange(0, (shape[1]+1)*spacing, 0.1)
    X, Y = np.meshgrid(X, Y)
    
    Z = scipy.interpolate.griddata((x,y), z, (X,Y), method='linear')
    
    max_pres = np.array(all_data).T[3].max()
    #print(max_pres)
    plt.contourf(X,Y,Z,cmap = cm.coolwarm,extend='both',vmin=0,vmax=max_pres)
    plt.title(name_file)
    
    num_cir = shape[0]*shape[1]
    for i in range(num_cir):
            x = spacing+spacing*(i % shape[0])
            y = spacing+spacing*(i % shape[1])
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),1/(2*shape[1]),fill=False,edgecolor='black'))
            #ax.add_patch(patches.Rectangle((x,y),-1/(2*shape[0]),-1/(2*shape[1]),fill=False,edgecolor='black'))
            ax.add_patch(patches.Circle((x,y),0.25,fill=True,edgecolor='red',facecolor='red'))
        