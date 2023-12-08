#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:52:06 2021

@author: pc-robotiq
"""

import os
import numpy as np
import pandas as pd
from math import sqrt,pi
import time

from ValidateFromFEM_roundedSurface import GaussianPressureDistribution, compare2FEM


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

#name_file = '-0,404226043915_-4,56785783664_173,430175862_17,3051882878_3,71580067543'
type_contact = 'SurfaceContact'
name_dir = "/home/pc-robotiq/measurements_FEM/"+type_contact+"/csvs/"
list_files = os.listdir(name_dir)
list_errors = np.array([])
list_time = []

if 'old' in list_files:
    list_files.remove('old')
if '0_0_18_36_0.csv' in list_files:
    list_files.remove('0_0_18_36_0.csv')

list_files_1 = list_files.copy()

for ind in range(len(list_files_1)):
    name_file = list_files_1[ind]
    if name_file[:6] == '.~lock':
        list_files.remove(name_file)

for ind in range(len(list_files)):
    #ind = 0
    #name_file = "/home/pc-robotiq/measurements_paper/FEM/csvs/"+name_file+".csv"    
    name_file = list_files[ind]
    full_name = name_dir+name_file
    print("------------------------------------------------")
    print(ind+1)
    #print(name_file)
    
    shape = [5,9]
    spacing = 4.5
    
    df=pd.read_csv(full_name, sep=',',header=None)
            
    data_raw_all = df.to_numpy()
    list_error_i = []
    
    for ti in range(1):
        #data_raw = data_raw.T[random.randint(2,len(data_raw[0])-1)]
        #data_raw = data_raw_all.T[ti]
        data_raw = data_raw_all.T[-1]
        t0 = time.time()
        
        
        results = np.zeros((4,1))
        
        pres = data_raw[1:46].copy()
        array, E = GaussianPressureDistribution(pres.copy(),shape,spacing,n=30,ftol=1e-2,max_it=5, verbose_bool=False)
        #print(E)
        dur = time.time()-t0
        
        #errorMatrix = calc_error(shape,spacing,E.x,name_file[:-4],type_contact)
        errorMatrix = compare2FEM(shape,spacing,E.x,name_file[:-4],type_contact,plot=True)
        #sprint("RMSE: "+str(sqrt(np.square(errorMatrix).mean())))
        #plot_roundedSurface(shape,spacing,E.x)
        F = calcForce_roundedSurface(E.x)
        F_ref = data_raw[46]
        #print("reference force: "+str(F_ref))
        print("force error: "+str(F-F_ref))
        
        
        #list_error_i.append(sqrt(np.square(errorMatrix).mean()))
        list_error_i.append((F-F_ref)/F_ref)
        
        
        print('duration: '+str(dur))
        list_time.append(dur)
    
    if len(list_errors) == 0:
        list_errors = np.append(list_errors,list_error_i)
    else:
        list_errors = np.vstack((list_errors,list_error_i))
    #time.sleep(15)



print(":::::::::::::::::::::::::::::::::::")
#print(sum(list_errors)/len(list_errors))
list_time = np.array(list_time)
print("max duration: "+str(max(list_time)))
print("RMSE force: "+str(sqrt((list_errors**2).mean())))
"""
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([0,27])
ax1.set_ylim([0,45])
    
ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([0,27])
ax2.set_ylim([0,45])
"""