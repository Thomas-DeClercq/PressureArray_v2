#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 13:36:31 2021

@author: pc-robotiq
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def conv(x):
    return x.replace(',', '.').encode()

data1 = []
name_dir = 'C-shape'
all_cases = os.listdir('./'+name_dir)
for case in all_cases:
    if '.' in case:
        all_cases.remove(case)
        
for i in range(len(all_cases)):
    if not (all_cases[i][-3:] == '.py' or all_cases[i][-4:] == '.txt' or all_cases[i] == 'old'):
        
        case = all_cases[i]
        data1 = []
        
        for i in range(45):
            raw_data = np.genfromtxt((conv(x) for x in open("./"+name_dir+"/"+case+"/sensor"+str(i+1)+".txt")),delimiter='\t',skip_header=1)
            raw_data = raw_data.T
            # raw_data[1] = raw_data[1]
            if i == 0:
                data1.append(raw_data[1][:])
            data1.append(-raw_data[4][:]*10**6) # np.append(data1,raw_data[4][:-2]*10**(-4))
            
        
            
        raw_force = raw_data.copy()
        raw_force = raw_force[1]*5
        
        data1 = np.array(data1)
        
        
        ind_2 = 0
        for i in range(data1.shape[1]):
            if data1[0][i] == 0:
                ind_2 = i
                break
        
        pressures0 = data1.T[ind_2][1:]
        
        #data2 = data1.T - np.append(0,pressures0)
        #data2 = data2.T
        data2 = data1
        data2[0] = data2[0] - data2[0][0]
        
        
        Fz = np.zeros((1,len(data2[0])))
        for i in range(len(raw_force)):
            Fz[0][i] = raw_force[i]
            
        case1 = case.split('/')
        x = float(case1[-1].split('_')[0].replace(',','.'))
        y = float(case1[-1].split('_')[1].replace(',','.'))
        r = float(case1[-1].split('_')[2].replace(',','.'))
        w= float(case1[-1].split('_')[3].replace(',','.'))
        
        x = x*np.ones((1,len(data2[0])))
        y = y*np.ones((1,len(data2[0])))
        r = r*np.ones((1,len(data2[0])))
        w = w*np.ones((1,len(data2[0])))
        
        data_csv = np.append(np.append(np.append(np.append(np.append(data2,Fz,axis=0),x,axis=0),y,axis=0),r,axis=0),w,axis=0)
        if len(case1) == 1:
            case_csv = case
        else:
            case_csv = case1[0]+"_"+case1[-1]
        name_csv = "./"+name_dir+"/"+case_csv+".csv"
        np.savetxt(name_csv,data_csv,delimiter=",")

"""
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(15):
    ax.plot(data2[0][ind_2:],data2[1+i][ind_2:],label="sensor "+str(i+1))
ax.legend()

fig1 = plt.figure()
axs = {}
for i in range(15):
    axs[i] = fig1.add_subplot(5,3,i+1)
    axs[i].plot(data2[0][ind_2:],data2[1+i][ind_2:])
    axs[i].title.set_text("sensor "+str(i+1))
""" 
