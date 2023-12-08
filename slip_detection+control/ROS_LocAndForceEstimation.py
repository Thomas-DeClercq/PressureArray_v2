#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:37:08 2021

@author: pc-robotiq
"""

import time
import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from std_msgs.msg import Bool

import _thread as thread

import math
from math import sqrt,pi,exp
import scipy
#from scipy.optimize import minimize, curve_fit
from minpack_adj import curve_fit
from scipy.signal import butter, filtfilt

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

def readPressures_t1(value):
    global pressure_data_t1
    for i in range(pressure_data_t1.shape[1]):
        pressure_data_t1[0][i] = value.data[i]
    return

def readPressures_t2(value):
    global pressure_data_t2
    for i in range(pressure_data_t2.shape[1]):
        pressure_data_t2[0][i] = value.data[i]
    #print(f"min t2: {np.min(pressure_data_t2)}; max t2: {np.max(pressure_data_t2)}")
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
        all_pressure_data[i] = pressure_data.copy()
        time.sleep(1/freq)
    
    pressure_data0 = np.average(all_pressure_data,axis=0)
    return

def update_data():
    global pressure_data

    pressure_data = np.append(pressure_data_t1.copy(),pressure_data_t2.copy())
    if wires_up:
        pressure_data = np.flip(pressure_data)
        
    return

def butter_lowpass_filter(data,on):
    if on:
        fs = freq
        cutoff = 5
        nyq = 0.5*fs
        order = 2
        
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff,btype='low', analog=False)
        # y = filtfilt(b,a,data)
        
        y = scipy.signal.lfilter(b,a,data)
    else:
        y = data
    return y

def std_gaussian(x,y,p0,std,x0,y0):
    return p0/(std*sqrt(2*pi))*exp(-0.5*(((x-x0)**2+(y-y0)**2)/(std**2)))

def gaussian(X,p0,std,x0,y0):
    xx = X[0]
    yy = X[1]
    for i,(x,y) in enumerate(zip(xx,yy)):
        X[2][i] = p0*exp(-0.5*(((x-x0)**2+(y-y0)**2)/(std**2)))
    return X[2]

def ObjFun(par,array,n): 
    error = 0
    #print(array)
    for i in range(n):
        #sigma = gaussian(array[i][2],array[i][3],par[0],par[1],par[2],par[3])
        
        sigma = gaussian(array[i][2],array[i][3],par[0],par[1],par[2],par[3])
        #sigma = gaussian(array[i][2],array[i][3],par[0],2,par[2],par[3])
        #print(sigma)
        #error += (len(array)-i)**2*abs(sigma-array[i][1])
        #print(abs(sigma-array[i][1]))
        #error += 1/array[i][1]*abs(sigma-array[i][1])
        error += abs(sigma-array[i][1])
        
    #error += (1/par[1])*10**-1.5
    #print(error)
    #print(par)        
    return error

def make_feasible(x0,lb,ub):
    for i,(x,lv, uv) in enumerate(zip(x0,lb,ub)):
        if x < lv:
            x0[i] = lv
        elif x > uv:
            x0[i] = uv
    return x0

def HerzianContactLoc(pressureArray, shape, spacing, t=6, n=9, x0=None, ftol=1e-3, max_it= 10, verbose_bool=False):
    """
    n: amount of points used
    """
    #global x0
    
    
    #t0 = time.time()
    amount = int(shape[0]*shape[1])
    n_p = min(n,amount)
    
    if verbose_bool:
        verbose = 2
    else:
        verbose = 0
    
    array = np.zeros((amount,4)) # n points with highest pressure; 1: sensor number; 2: pressure; 3: x coordinate; 4: y coordinate
    
    for i in range(amount):
        array[i][0] = np.argmax(pressureArray)
        array[i][1] = pressureArray[int(array[i][0])]
        
        pressureArray[int(array[i][0])] = -10**9
        
    for i in range(amount):
        idx = array[i][0]
        x = spacing + spacing*(idx % shape[0])
        y = spacing*(shape[1]) - spacing*math.floor(idx/shape[0])
        array[i][2] = x
        array[i][3] = y
         
    #print(array)    
    
    #x0 = [array[0][1],2,array[0][2],array[0][3]] # [k, a, x_c, y_c]
    #bnds = scipy.optimize.Bounds([array[0][1],0,array[0][2]-spacing,array[0][3]-spacing],[float('inf'),float('inf'),array[0][2]-spacing,array[0][3]-spacing])
    """
    if not success:
        x0 = [array[0][1],1,array[0][2],array[0][3]] # [k, x_c, y_c]
    """
    #x0 = [array[0][1]/gaussian(0,0,1,1,0,0),1,array[0][2],array[0][3]]
    #bnds = scipy.optimize.Bounds([array[0][1]//gaussian(0,0,1,1,0,0),0,min(array[0][2]-spacing,0),min(array[0][3]-spacing,0)],[float('inf'),float('inf'),max(array[0][2]+spacing,shape[0]*spacing),max(array[0][3]+spacing,shape[0]*spacing)])
    
    
    ### normalize
    length_0 = max(shape[0]*spacing,shape[1]*spacing)
    array_0 = np.zeros(array.shape)
    array_0.T[0] = array.T[0]
    array_0.T[1] = array.T[1].T/np.max(array.T[1])
    array_0.T[2] = array.T[2]/(length_0)
    array_0.T[3] = array.T[3]/(length_0)
    
        #print(array_0)
    n0 = 0
    for i0 in range(len(array_0)):
        if array_0[i0][1] > 0.4:
            n0 += 1
    
    #print(n0)
    diff_x = False
    diff_y = False
    
    array_n = array[:n_p]
    
    n = 3
    nx = []
    ny = []
    while not (diff_x and diff_y):
        for i in range(n):
            for j in range(i):
                if (array[i][2] != array[j][2]) and (not diff_x):
                    diff_x = True
                    nx.append(j)
                    nx.append(i)
                if (array[i][3] != array[j][3]) and (not diff_y):
                    diff_y = True
                    ny.append(j)
                    ny.append(i)
        if not (diff_x and diff_y):
            n += 1
    
    # x0 = [p0, std, x0, y0]
    
    if x0 is None:    
        x0 = [array_n[0][1],3.5,(array_n.T[2][nx[0]]+array_n.T[2][nx[1]])/2,(array_n.T[3][ny[0]]+array_n.T[3][ny[1]])/2]
    #x0 = [array_n[0][1],0.1,array_n[0][2],array_n[0][3]]
    
    #print(x0)
    """
    bnds = scipy.optimize.Bounds([x0[0],0.0,array_n.T[2][0:nx].min(),array_n.T[3][0:nx].min()],[float('inf'),float('inf'),array_n.T[2][0:nx].max(),array_n.T[3][0:nx].max()])
    
    
    lb = [x0[0], 0, min(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), min(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    ub = [x0[0]*100, 10, max(array_n.T[2][nx[0]],array_n.T[2][nx[1]]), max(array_n.T[3][ny[0]],array_n.T[3][ny[1]])]
    """
    lb = [x0[0], 0, max(array_n.T[2][0]-spacing,spacing/2), max(array_n.T[3][0]-spacing, spacing/2)]
    ub = [x0[0]*100+1, 10, min(array_n.T[2][0]+spacing, shape[0]*spacing-spacing/2), min(array_n.T[3][0]+spacing,shape[1]*spacing-spacing/2)]
    """
    print(lb)
    print(ub)
    print('-------------------------')
    """
    x0 = make_feasible(x0,lb,ub)
    
    xdata = array_n.T[2:].T[:n_p].T
    xdata = np.append(xdata,np.zeros((1,xdata.shape[1])),axis=0)
    ydata = array_n.T[1][:n_p]
    
    #print(xdata)
    #print(ydata)
    meth = ['trf','dogbox']
    
    E = curve_fit(gaussian, xdata, ydata, x0, bounds=(lb, ub), method = meth[0], maxfev=max_it, verbose=verbose,ftol=ftol)
    """
    if E.success:
        x0 = E.x
    """
    #E.x[0] = E.x[0]/(E.x[1])**2 #==> E.x = [p0,z,x_c,y_c] p0 in Pa
    #print(E)
    #print(1/(time.time()-t0))
    #print(E)
    return array,E

def SlipDetectionDist(array_x,array_y,frac=0.1,dist_thres=0.1,use_last=-1):
    if len(array_x) == 0 or len(array_y) == 0:
        return False,0,0
    
    if use_last != -1:
        array_x = array_x[-use_last:]
        array_y = array_y[-use_last:]
    
    ind = int(len(array_x)*frac)
    
    x_mean = sum(array_x[:ind])/len(array_x[:ind])
    y_mean = sum(array_y[:ind])/len(array_y[:ind])
    
    new_x_mean = sum(array_x[ind:])/len(array_x[ind:])
    new_y_mean = sum(array_y[ind:])/len(array_y[ind:])
    
    dist = sqrt((x_mean-new_x_mean)**2+(y_mean-new_y_mean)**2)
    
    theta = math.atan2(array_y[-1]-y_mean,array_x[-1]-x_mean)/pi*180
    #print(theta/pi*180)
    #print(dist)
    #theta = np.arctan((new_y_mean-y_mean)/(new_x_mean-x_mean))/pi*180
    #print(theta)
    
    if dist > dist_thres:
        return True, dist, theta
    else:
        return False, dist, theta
       
if __name__ == '__main__':
    rospy.init_node('LocAndForceEstimation')
    freq = 20
    k = 2*pi*1e-6
    shape = [4,8]
    spacing = 4.5 # m
    wires_up = False

    pressure_data_t1 = np.zeros((1,int(shape[0]*shape[1]/2)))
    pressure_data_t2 = np.zeros((1,int(shape[0]*shape[1]/2)))
    pressure_data = np.zeros((1,int(shape[0]*shape[1])))
    pressure_data0 = np.zeros((1,int(shape[0]*shape[1])))
    x_sample = []

    y_sample = []
    x_sample_fund = []
    y_sample_fund = []
    t_start = 0

    success = False
    x0 = []
    coord = Float64MultiArray()

    for i in range(freq*1):
        x_sample.append(0)
        y_sample.append(0)
        
    calibrated = False
    rate = rospy.Rate(freq)
    
    pub_F = rospy.Publisher('estimatedForce',Float64,queue_size=1)
    pub_Loc = rospy.Publisher('estimatedLocation',Float64MultiArray,queue_size=1)
    pub_SLIP = rospy.Publisher('SLIP',Bool,queue_size=1)
    pub_SLIP_speed = rospy.Publisher('SLIP_speed',Float64,queue_size=1)
    pub_SLIP_angle = rospy.Publisher('SLIP_angle',Float64,queue_size=1)
    rospy.Subscriber('/Pressure_t1',Float64MultiArray,readPressures_t1,queue_size=1)
    rospy.Subscriber('/Pressure_t2',Float64MultiArray,readPressures_t2,queue_size=1)

    calibrated = False
    

    update_data()
    time.sleep(0.5)
    update_data()
    set_pressure0()
    calibrated = True
    rate.sleep()
    update_data()
    
    thread.start_new_thread(manual_recalibration,())
    
    while not rospy.is_shutdown():
        #update_data()
        t0 = time.time()
        pressures = []
        update_data()
        """
        for i in range(len(rec)):
            rec["rec"+str(i+1)].xy = (-10,-10)
        """
    
        for i in range(pressure_data.shape[0]):
                pressures.append(pressure_data[i]-pressure_data0[i])
                
        #print(f"{np.max(pressures)} @ {np.argmax(pressures)}")
        #print(pressures)
        if np.max(pressures) > 1000:
            #print(np.max(pressures))
            array, E = HerzianContactLoc(pressures, shape, spacing, n=9, x0=None, ftol=1e-2, max_it= 10, verbose_bool=False)
            
            #print(array)
            success = E.success
            result = E.x
            #print(result)
            
            if E.success:
                
                if (len(x_sample_fund) < len(x_sample)) and (len(y_sample_fund) < len(y_sample)):
                    x_sample_fund.append(result[2])
                    y_sample_fund.append(result[3])
                    
                    n = math.floor(len(x_sample)/len(x_sample_fund))
                    for idx in range(len(x_sample)):
                        idxx = math.floor(idx/n)
                        if idxx >= len(x_sample_fund):
                            idxx = len(x_sample_fund)-1
                        x_sample[idx] = x_sample_fund[idxx]
                        y_sample[idx] = y_sample_fund[idxx]
                        
                else:
                    _ = x_sample_fund.pop(0)
                    x_sample_fund.append(result[2])
                    _ = y_sample_fund.pop(0)
                    y_sample_fund.append(result[3])
                    
                    x_sample = x_sample_fund.copy()
                    y_sample = y_sample_fund.copy()
                
                x_f = butter_lowpass_filter(x_sample,False)
                y_f = butter_lowpass_filter(y_sample,False)
                
                x_sample[-1] = x_f[-1]
                y_sample[-1] = y_f[-1]
                x_sample_fund[-1] = x_f[-1]
                y_sample_fund[-1] = y_f[-1]
                
                slip,slip_speed,slip_angle = SlipDetectionDist(x_sample,y_sample,frac=0.34,dist_thres=0.75,use_last=3)

                """
                if slip:
                    print("SLIP")
                """
                pub_SLIP.publish(slip)
                pub_SLIP_speed.publish(slip_speed)
                pub_SLIP_angle.publish(slip_angle)
                
                #F = k*E.x[0]*E.x[1]
                F = k*E.x[0]*E.x[1]**2
                pub_F.publish(-F)
                coord.data = [x_f[-1],y_f[-1]]
                #print(coord.data)
                #print(F)
                pub_Loc.publish(coord)

                
                calibrated = False

        else:
            ti = time.time()
            success = False
            
            """
            if not False: #calibrated:
                set_pressure0()
                calibrated = True
            """
            for i in range(len(x_sample)):
                _ = x_sample.pop(0)
                x_sample.append(0)
                _ = y_sample.pop(0)
                y_sample.append(0)
                
                x_sample_fund = []
                y_sample_fund = []
                
            F = 0
            pub_F.publish(-F)
            coord.data = [-1, -1]
            pub_Loc.publish(coord)

            pub_SLIP.publish(False)
            pub_SLIP_speed.publish(-999)
            pub_SLIP_angle.publish(-999)

            #TXT.set_x = result[1]
            #TXT.set_y = result[2]
            
            
        rate.sleep()
        
    """
    thread.start_new_thread(program,())
    thread.start_new_thread(manual_recalibration,())
    while not rospy.is_shutdown():
        time.sleep(1)
    """
