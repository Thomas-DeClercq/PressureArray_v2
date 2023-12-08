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

from ValidateFromFEM_roundedSurface import GaussianPressureDistribution, compare2FEM, calcForce_int

pressure_data = np.zeros((1,9));
pressure_data0 = np.zeros((1,9));

success = False
x0 = []

freq = 33

params = Float64MultiArray()
    
calibrated = False


def readTopic(value):
    global pressure_data
    for i in range(pressure_data.shape[1]):
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
    
if __name__ == '__main__':
    rospy.init_node('LocAndForceEstimation')
    rate = rospy.Rate(freq)
    
    pub_F = rospy.Publisher('estimatedForce',Float64,queue_size=1)
    pub_par = rospy.Publisher('estimatedParameters',Float64MultiArray,queue_size=1)
    
    calibrated = False
    shape = [3,3]
    spacing = 4 # mm

    update_data()
    time.sleep(0.5)
    update_data()
    set_pressure0()
    calibrated = True
    rate.sleep()
    update_data()
    x0 = None
    
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
        #print("x0= "+str(x0))
        if np.max(pressures) > 1000:
            #print(np.max(pressures))
            array, E = GaussianPressureDistribution(pressures, shape, spacing, t=6, n=9, x0=x0, ftol=1e-2, verbose_bool=False, max_it=10)
            
            #print(array)
            success = E.success
            #print(E.x)
            params.data = [*E.x]
            #print(params)
            
            if E.success:

                F = calcForce_roundedSurface(params.data)
                F_int = calcForce_int(params.data,shape,spacing,n=10)
                pub_F.publish(-F_int)
                #print(F)
                pub_par.publish(params)
                #print(result)
                calibrated = False
                #x0 = [*E.x]

        else:
            success = False
            
            if not calibrated:
                set_pressure0()
                calibrated = True
                
            #F = 0
            pub_F.publish(0)
            params.data = [0,0,0,0,0,-1,-1]
            pub_par.publish(params)
            x0 = None
            
        #print(1/(time.time()-t0))
        rate.sleep()
