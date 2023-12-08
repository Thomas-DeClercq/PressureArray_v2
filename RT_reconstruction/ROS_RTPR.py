#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:26:52 2023

@author: pc-robotiq
"""
import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from std_msgs.msg import Bool

import numpy as np
import scipy
import time

import _thread as thread

from scipy.interpolate import RBFInterpolator, Rbf

import sys
sys.path.insert(1,'/home/thomas/pythonScripts/PressureArray_v2/PR_cython')
#from PressureReconstruction_onlyPoints import Optimization, calc_Z
#from PressureReconstruction_update210623 import Optimization, calc_Z
from PressureReconstruction_angleEstimation import Optimization, calc_Z

def readPressures_t1(value):
    global pressure_data_t1
    for i in range(pressure_data_t1.shape[1]):
        pressure_data_t1[0][i] = value.data[i]
    return

def readPressures_t2(value):
    global pressure_data_t2
    for i in range(pressure_data_t2.shape[1]):
        pressure_data_t2[0][i] = value.data[i]
    return

def manual_recalibration():
    while not rospy.is_shutdown():
        text = input("Press 0 to recalibrate pressure sensors\n")
        if text == '0':
            set_pressure0()
        else:
            print("input is not 0")
        time.sleep(0.1)
    return
    
def set_pressure0():
    global pressure_data0
    update_data()

    if np.sum(pressure_data) == 0:
        time.sleep(1/freq)

    all_pressure_data = np.zeros((freq,pressure_data.shape[0]))
    for i in range(freq): #during 1s
        update_data()
        #print(pressure_data)
        all_pressure_data[i] = pressure_data
        time.sleep(1/freq)
    
    pressure_data0 = np.average(all_pressure_data,axis=0)
    return
        

def update_data():
    global pressure_data

    pressure_data = np.append(pressure_data_t1,pressure_data_t2)

    if wires_up:
        pressure_data = np.flip(pressure_data)
        
    return

def getValue(X,Y,Z,xi,yi):
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    point = list(zip(xi,yi))
    #point = [(xi,yi)]
    zi = scipy.interpolate.griddata((X,Y),Z,point)
    zi = np.nan_to_num(zi)
    return zi
    
def integratePressure(X_ls,Y_ls,X,Y,params):
    Z = np.zeros(X.flatten().shape)
    
    it = int(params.shape[0]/8)
    for idx in range(it):
        try:
            Z_add = calc_Z(X.flatten(),Y.flatten(),*params[idx*8:(idx+1)*8])
        except Exception as e:
            Z_add = np.zeros(X.flatten().shape)
            print(f"{e} with parameters: {params[idx]}")
        Z = Z + Z_add
    Z = np.reshape(Z, X.shape)
    
    Z[Z<0] = 0 #pressure cannot be negative
    
    return scipy.integrate.trapz([scipy.integrate.trapz(Z_x,X_ls) for Z_x in Z],Y_ls)

if __name__ == '__main__':
    freq = 20
    rospy.init_node('LocAndForceEstimation')
    rate = rospy.Rate(freq)
    
    pub_par = rospy.Publisher('estimatedParameters',Float64MultiArray,queue_size=1)
    rospy.Subscriber('/Pressure_t1',Float64MultiArray,readPressures_t1,queue_size=1)
    rospy.Subscriber('/Pressure_t2',Float64MultiArray,readPressures_t2,queue_size=1)
    params = Float64MultiArray()
    pub_force = rospy.Publisher('estimatedForce',Float64,queue_size=1)
    force = Float64()

    shape = [4,8]
    spacing = 4.5

    pressure_data_t1 = np.zeros((1,int(shape[0]*shape[1]/2)))
    pressure_data_t2 = np.zeros((1,int(shape[0]*shape[1]/2)))
    pressure_data = np.zeros((1,int(shape[0]*shape[1])))
    pressure_data0 = np.zeros((1,int(shape[0]*shape[1])))

    wires_up = True

    n = 50
    m = 20 # keep between 10 - 15
    p_m = 0.75
    max_it = 5
    max_t = 45
    
    list_time = []
    xi = []
    yi = []
    
    for y_i in range(shape[1]):
        for x_i in range(shape[0]):
            xi.append(spacing+x_i*spacing)
            yi.append(spacing*(shape[1])-y_i*spacing)

            
    xi = np.array(xi)
    yi = np.array(yi)
            
    RMSE = np.array([])
    RMSE_rbf = np.array([])
    times = np.array([])
    
    params_names = ['p0','std','lx','ly','r_curve','theta','x0','y0']
    X_ls = np.linspace(0, (shape[0]+1)*spacing, n)
    Y_ls = np.linspace(0, (shape[1]+1)*spacing, n)
    X, Y = np.meshgrid(X_ls, Y_ls)
    
    X_rbf = np.linspace(0, (shape[0]+1)*spacing, m)
    Y_rbf = np.linspace(0, (shape[1]+1)*spacing, m)
    X_rbf, Y_rbf = np.meshgrid(X_rbf, Y_rbf)
    
    list_x0 = np.zeros((max_it,8))

    update_data()
    set_pressure0()
    calibrated = True
    rate.sleep()

    thread.start_new_thread(manual_recalibration,())
    
    while not rospy.is_shutdown():
        t0 = time.time()
        pressures = []
        update_data()
        for i in range(pressure_data.shape[0]):
                pressures.append(pressure_data[i]-pressure_data0[i])

        if np.max(pressures) > 5000:
            rbfi = Rbf(xi,yi,pressures,function='gaussian') #always +-2.5 ms
            t2 = time.time()
            Z_rbf = rbfi(X_rbf,Y_rbf)
            array_rbf = np.array([X_rbf.flatten(),Y_rbf.flatten(),Z_rbf.flatten()]).T
            t3 = time.time()
            idxs = []
            for idx,el in enumerate(array_rbf):
                if el[0] < spacing or el[0] > shape[0]*spacing:
                    idxs.append(idx)
                elif el[1] < spacing or el[1] > shape[1]*spacing:
                    idxs.append(idx)
            array_rbf = np.delete(array_rbf,idxs,axis=0).T
            
            pressures = np.array(pressures)
            array, list_E = Optimization(xi,yi, pressures.copy(),shape,spacing, array_rbf, list_x0, n=round(p_m*array_rbf.shape[1]),it_max=max_it,t_max=max_t)

            par_array = np.array([])
            for i,E in enumerate(list_E):
                    par_array = np.append(par_array,[E.x])
                    list_x0[i] = E.x
            params.data = par_array
            k = 10**(-6)
            estimated_force = integratePressure(X_ls,Y_ls,X,Y,par_array)
            estimated_force = k*estimated_force

        else:
            params.data = [0]*8
            params.data[1] = 1
            list_x0 = np.zeros((max_it,8))
            estimated_force = 0
            
        #print(params)
        pub_force.publish(estimated_force)
        pub_par.publish(params)
        rate.sleep()
