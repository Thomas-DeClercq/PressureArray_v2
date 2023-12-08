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

import math
from math import sqrt,pi,exp
import scipy
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation


pressure_data = np.zeros((1,15));
pressure_data0 = np.zeros((1,15));
x_sample = []
y_sample = []
x_sample_fund = []
y_sample_fund = []
t_start = 0

success = False
x0 = []

freq = 30

coord = Float64MultiArray()

k = 0.001272164923549179 # from CalibrationForce_All.py

for i in range(freq*1):
    x_sample.append(0)
    y_sample.append(0)
    
calibrated = False


def readTopic(value):
    global pressure_data
    for i in range(15):
        pressure_data[0][i] = value.data[i]
    return
    
def set_pressure0():
    global pressure_data0
    for i in range(pressure_data0.shape[1]):
        pressure_data0[0][i] = pressure_data[0][i]
        


def update_data():
    global t
    global t0
    global pressure_data

    rospy.Subscriber('/Pressure',Float64MultiArray,readTopic,queue_size=1)
    
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

def gaussian(x,y,p0,std,x0,y0):
    return p0*exp(-0.5*(((x-x0)**2+(y-y0)**2)/(std**2)))

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

def HerzianContactLoc(pressureArray, shape, spacing, t=6,n=9):
    """
    n: amount of points used
    """
    global x0
    
    # t0 = time.time()
    amount = int(shape[0]*shape[1])
    
    array = np.zeros((amount,4)) # n points with highest pressure; 1: sensor number; 2: pressure; 3: x coordinate; 4: y coordinate
    
    for i in range(amount):
        array[i][0] = np.argmax(pressureArray)
        array[i][1] = pressureArray[int(array[i][0])]
        
        pressureArray[int(array[i][0])] = -10**9
        
    for i in range(amount):
        idx = array[i][0]
        x = spacing/2 + spacing*(idx % shape[0])
        y = spacing*shape[1] - (spacing/2 + spacing*math.floor(idx/shape[0]))
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
    length_n = max(shape[0]*spacing,shape[1]*spacing)
    array_n = np.zeros(array.shape)
    array_n.T[0] = array.T[0]
    array_n.T[1] = array.T[1].T/np.max(array.T[1])
    array_n.T[2] = array.T[2]/(length_n)
    array_n.T[3] = array.T[3]/(length_n)
    
    #print(array_n)
    """
    if not success:
        x0 = [array_n[0][1],0.1,array_n[0][2],array_n[0][3]]
        
    else:
        x0[0] = x0[0]/np.max(array.T[1])
        x0[1] = 0.1
        x0[2] = x0[2]/(length_n)
        x0[3] = x0[3]/(length_n)
    """
    

    
    ## have to adjust the bounds due to new normalization constant length_n
    #bnds = scipy.optimize.Bounds([x0[0],0,max(array_n[0][2]-1/shape[0],0),max(array_n[0][3]-1/shape[1],0)],[float('inf'),float('inf'),min(array_n[0][2]+1/shape[0],1),min(array_n[0][3]+1/shape[1],1)])   
    #bnds = scipy.optimize.Bounds([x0[0],0,max(array_n[0][2]-1/(2*shape[0]),0),max(array_n[0][3]-1/(2*shape[1]),0)],[float('inf'),float('inf'),min(array_n[0][2]+1/(2*shape[0]),1),min(array_n[0][3]+1/(2*shape[1]),1)])   
    #bnds = scipy.optimize.Bounds([x0[0],0,max(array_n[0][2]-1/(2*shape[0]),1/(2*shape[0])),max(array_n[0][3]-1/(2*shape[1]),1/(2*shape[1]))],[float('inf'),float('inf'),min(array_n[0][2]+1/(2*shape[0]),1-1/(2*shape[0])),min(array_n[0][3]+1/(2*shape[1]),1-1/(2*shape[1]))])   
    #bnds = scipy.optimize.Bounds([x0[0],0,max(array_n[0][2]-1/(shape[0]),1/(2*shape[0])),max(array_n[0][3]-1/(shape[1]),1/(2*shape[1]))],[float('inf'),float('inf'),min(array_n[0][2]+1/(shape[0]),1-1/(2*shape[0])),min(array_n[0][3]+1/(shape[1]),1-1/(2*shape[1]))])   

    diff_x = False
    diff_y = False
    """
    for i in range(n):
        for j in range(i):
            if array[i][2] != array[j][2]:
                diff_x = True
            if array[i][3] != array[j][3]:
                diff_y = True
                
    if diff_x and diff_y:
        idx = n
    else:
        idx = n + 1
    """
    nx = 3
    while (not diff_x) and (not diff_y):
        for i in range(nx):
            for j in range(i):
                if array[i][2] != array[j][2]:
                    diff_x = True
                if array[i][3] != array[j][3]:
                    diff_y = True
        if not (diff_x and diff_y):
            nx += 1
    
    #print(nx)
    x0 = [array_n[0][1],0.1,(array_n.T[2][0:nx].max()+array_n.T[2][0:nx].min())/2,(array_n.T[3][0:nx].max()+array_n.T[3][0:nx].min())/2]
    #x0 = [array_n[0][1],0.1,array_n[0][2],array_n[0][3]]
    
    #print(x0)
    
    bnds = scipy.optimize.Bounds([x0[0],0.0,array_n.T[2][0:nx].min(),array_n.T[3][0:nx].min()],[float('inf'),float('inf'),array_n.T[2][0:nx].max(),array_n.T[3][0:nx].max()])
    #print(bnds)
    
    meth = ['Nelder-Mead','L-BFGS-B','Powell','TNC','SLSQP']
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    opt = {}
    opt_0 = {'maxfev':10**3}
    opt_1 = {'ftol':1e-9,'gtol':1e-7,'maxls':40}
    opt_4 = {'ftol':1e-12,'maxiter':2*10**4}
    
    E = minimize(ObjFun,x0,args=(array_n,n),method = meth[0],bounds=bnds,options=opt_0)
    
    ### denormalize
    E.x[0] = E.x[0]*np.max(array.T[1])
    E.x[2] = E.x[2]*length_n
    E.x[3] = E.x[3]*length_n
    """
    if E.success:
        x0 = E.x
    """
    #E.x[0] = E.x[0]/(E.x[1])**2 #==> E.x = [p0,z,x_c,y_c] p0 in Pa
    #print(E)
    #print(1/(time.time()-t0))
    return array,E

def SlipDetectionDist(array_x,array_y,frac=0.1,dist_thres=0.1,use_last=-1):
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
    rate = rospy.Rate(freq)
    
    pub_F = rospy.Publisher('estimatedForce',Float64,queue_size=1)
    pub_Loc = rospy.Publisher('estimatedLocation',Float64MultiArray,queue_size=1)
    pub_SLIP = rospy.Publisher('SLIP',Bool,queue_size=1)
    pub_SLIP_speed = rospy.Publisher('SLIP_speed',Float64,queue_size=1)
    pub_SLIP_angle = rospy.Publisher('SLIP_angle',Float64,queue_size=1)
    
    calibrated = False
    shape = [3,5]
    spacing = 9 # mm

    update_data()
    time.sleep(0.5)
    update_data()
    set_pressure0()
    calibrated = True
    rate.sleep()
    update_data()
    
    while not rospy.is_shutdown():
        #update_data()
        t0 = time.time()
        pressures = []
        """
        for i in range(len(rec)):
            rec["rec"+str(i+1)].xy = (-10,-10)
        """
        for i in range(pressure_data.shape[1]):
                pressures.append(pressure_data[0][i]-pressure_data0[0][i])
                
        #print(pressures)
        if np.max(pressures) > 1000:
            #print(np.max(pressures))
            array, E = HerzianContactLoc(pressures, shape, spacing)
            
            #print(array)
            success = E.success
            result = E.x
            
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
                
                slip,slip_speed,slip_angle = SlipDetectionDist(x_sample,y_sample,frac=0.2,dist_thres=1.2,use_last=15)

                """
                if slip:
                    print("SLIP")
                """
                pub_SLIP.publish(slip)
                pub_SLIP_speed.publish(slip_speed)
                pub_SLIP_angle.publish(slip_angle)
                
                F = k*E.x[0]*E.x[1]
                pub_F.publish(F)
                coord.data = [x_f[-1],y_f[-1]]
                #print(coord.data)
                #print(F)
                pub_Loc.publish(coord)

                
                calibrated = False

        else:
            success = False
            
            if not False: #calibrated:
                set_pressure0()
                calibrated = True
            
            for i in range(len(x_sample)):
                _ = x_sample.pop(0)
                x_sample.append(0)
                _ = y_sample.pop(0)
                y_sample.append(0)
                
                x_sample_fund = []
                y_sample_fund = []
                
            F = 0
            pub_F.publish(-F)
            pub_SLIP.publish(False)
            #TXT.set_x = result[1]
            #TXT.set_y = result[2]
            
        rate.sleep()
