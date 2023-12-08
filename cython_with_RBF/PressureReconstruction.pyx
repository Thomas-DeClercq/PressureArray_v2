#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 15:09:43 2022

@author: pc-robotiq
"""
import cython
cimport cython
from cython.parallel import prange,parallel

import numpy as np
cimport numpy as np

from cpython cimport array
ctypedef np.float_t DTYPE_t
ctypedef np.long_t DTYPE_l
ctypedef np.int64_t DTYPE_i
from libc.math cimport fabs, fmin

#from math import sqrt,pi,exp,sin,cos
import math
from libc.math cimport sqrt, pi, exp, sin, cos, atan2

#from libc.time cimport time, time_t

import scipy
from scipy.optimize import Bounds, curve_fit
from minpack_adj import curve_fit

import time
    
cdef float gaussian_r(double r, double p0, double std):
    return p0*exp(-0.5*(r/std)**2)

cpdef np.ndarray[DTYPE_t,ndim=1] calc_Z(np.ndarray[DTYPE_t,ndim=1] X, np.ndarray[DTYPE_t,ndim=1] Y, float p0, float std, float lx,float ly, float r_curve1, float theta1, float x0,float y0):    
    cdef float theta = theta1/180*pi
    cdef Py_ssize_t n = len(X)
    
    cdef np.ndarray[DTYPE_t,ndim=1] Z = np.zeros((len(X)))

    cdef float r_curve
    cdef signed int i
    cdef float x,y,dist
    cdef float x_theta, y_theta, r, alpha
    
    if r_curve1 < -0.1:
        r_curve = -10/(10**(-r_curve1-1))
        for i in range(n):
            x_theta = (X[i]-x0)*cos(theta)+(Y[i]-y0)*sin(theta)
            y_theta = -(X[i]-x0)*sin(theta)+(Y[i]-y0)*cos(theta)
            r = sqrt(x_theta**2+(y_theta-r_curve)**2)
            alpha = atan2(x_theta,y_theta-r_curve)
            x = alpha*r
            y = r+r_curve
            if (fabs(x) < lx/2 and fabs(y) < ly/2):
                Z[i] = p0
            elif (fabs(x) > lx/2 and fabs(y) > ly/2):
                r = fmin((x-lx/2)**2+(y-ly/2)**2,
                    fmin((x+lx/2)**2+(y-ly/2)**2,
                    fmin((x-lx/2)**2+(y+ly/2)**2,
                    (x+lx/2)**2+(y+ly/2)**2)))
                r = sqrt(r)
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif x < -lx/2:
                r = -lx/2-x
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif x > lx/2:
                r = x-lx/2
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif y < -ly/2:
                r = -ly/2-y
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif y > ly/2:
                r = y-ly/2
                Z[i] = p0*exp(-0.5*(r/std)**2)
            else:
                Z[i] = p0
    elif r_curve1 > 0.1:
        r_curve = 10/(10**(r_curve1-1))
        for i in range(n):
            x_theta = (X[i]-x0)*cos(theta)+(Y[i]-y0)*sin(theta)
            y_theta = -(X[i]-x0)*sin(theta)+(Y[i]-y0)*cos(theta)
            r = sqrt(x_theta**2+(y_theta-r_curve)**2)
            alpha = atan2(x_theta,-(y_theta-r_curve))
            x = alpha*r
            y = r - r_curve
            if (fabs(x) < lx/2 and fabs(y) < ly/2):
                Z[i] = p0
            elif (fabs(x) > lx/2 and fabs(y) > ly/2):
                r = fmin((x-lx/2)**2+(y-ly/2)**2,
                    fmin((x+lx/2)**2+(y-ly/2)**2,
                    fmin((x-lx/2)**2+(y+ly/2)**2,
                    (x+lx/2)**2+(y+ly/2)**2)))
                r = sqrt(r)
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif x < -lx/2:
                r = -lx/2-x
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif x > lx/2:
                r = x-lx/2
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif y < -ly/2:
                r = -ly/2-y
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif y > ly/2:
                r = y-ly/2
                Z[i] = p0*exp(-0.5*(r/std)**2)
            else:
                Z[i] = p0
    else:
        for i in range(n):
            x = (X[i]-x0)*cos(theta)+(Y[i]-y0)*sin(theta)
            y = -(X[i]-x0)*sin(theta)+(Y[i]-y0)*cos(theta)
            if (fabs(x) < lx/2 and fabs(y) < ly/2):
                Z[i] = p0
            elif (fabs(x) > lx/2 and fabs(y) > ly/2):
                r = fmin((x-lx/2)**2+(y-ly/2)**2,
                    fmin((x+lx/2)**2+(y-ly/2)**2,
                    fmin((x-lx/2)**2+(y+ly/2)**2,
                    (x+lx/2)**2+(y+ly/2)**2)))
                r = sqrt(r)
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif x < -lx/2:
                r = -lx/2-x
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif x > lx/2:
                r = x-lx/2
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif y < -ly/2:
                r = -ly/2-y
                Z[i] = p0*exp(-0.5*(r/std)**2)
            elif y > ly/2:
                r = y-ly/2
                Z[i] = p0*exp(-0.5*(r/std)**2)
            else:
                Z[i] = p0

    return Z

cdef check_feasible(list x0, list lb, list ub):
    cdef int i
    for i in range(len(x0)):
        if x0[i] > ub[i] or x0[i] < lb[i]:
            return False
    return True


cdef find_best_x0(np.ndarray[DTYPE_t,ndim=2] list_x0, list lb, list ub, float s):
    cdef unsigned int it
    cdef list matching_s = []
    cdef list matching = []
    cdef list x0
    
    for it in range(len(list_x0)):
        if (list_x0[it][0] > lb[0]) and (list_x0[it][0] < ub[0]) and (list_x0[it][6] > lb[6]-s) and (list_x0[it][6] < ub[6]+s) and (list_x0[it][7] > lb[7]-s) and (list_x0[it][7] < ub[7]+s):
            #check if near 
            matching_s.append(it)
            if (list_x0[it][6] > lb[6]) and (list_x0[it][6] < ub[6]) and (list_x0[it][7] > lb[7]) and (list_x0[it][7] < ub[7]):
                matching.append(it)
                
    
    cdef unsigned int amount = len(matching)
    cdef unsigned int amount_s = len(matching_s)
    cdef float p0_bounds
    if amount == 1:
        x0 = list(list_x0[matching[0]])
        list_x0[matching[0]] = np.zeros((8,))
    elif amount > 1:
        best_matching = matching[0]
        p0_bounds = lb[0]*ub[0]/1.5
        for it in matching:
            if (abs(list_x0[it][0])-p0_bounds < abs(list_x0[best_matching][0])-p0_bounds):
                best_matching = it      
        x0 = list(list_x0[best_matching])
        list_x0[best_matching] = np.zeros((8,))
    elif amount_s == 1:
        x0 = list(list_x0[matching_s[0]])
        lb[6] = lb[6]-s
        ub[6] = ub[6]+s
        lb[7] = lb[7]-s
        ub[7] = ub[7]+s
        list_x0[matching_s[0]] = np.zeros((8,))
    elif amount_s > 1:
        best_matching = matching_s[0]
        p0_bounds = lb[0]*ub[0]/1.5
        for it in matching_s:
            if (abs(list_x0[it][0])-p0_bounds < abs(list_x0[best_matching][0])-p0_bounds):
                best_matching = it
        x0 = list(list_x0[best_matching])
        lb[6] = lb[6]-s
        ub[6] = ub[6]+s
        lb[7] = lb[7]-s
        ub[7] = ub[7]+s
        list_x0[best_matching] = np.zeros((8,))
    else:
        x0 = []
    return x0, lb, ub, list_x0


cdef goodGuess(np.ndarray[DTYPE_t,ndim=2] array, list shape, float spacing, np.ndarray[DTYPE_t,ndim=2] list_x0):
    #params_names = ['p0','std','lx','ly','r_curve','theta','x0','y0'] #['p0','std','lx','ly','S_x','S_y','S','r_curve','F','theta','x0','y0']
    
    #print(array)
    #normalize pressure
    
    cdef np.ndarray[DTYPE_t,ndim=2] array_0 = np.zeros(np.shape(array))
    #array_0.T[0] = array.T[0]
    if array[0][0] > -1*array[-1][0]:
        array_0.T[0] = array.T[0].T/array[0][0]
        array_0.T[1] = array.T[1]
        array_0.T[2] = array.T[2]
        p_x0 = array[0][0]
        
    else:
        array_0.T[0] = array.T[0].T/array[-1][0]
        array_0.T[1] = np.flip(array.T[1])
        array_0.T[2] = np.flip(array.T[2])
        p_x0 = array[-1][0]
    
    #print(array_0)
    cdef unsigned int n0 = 0
    cdef unsigned int i0,i
    
    for i0 in range(len(array_0)):
        if array_0[i0][0] > 0.9:
            n0 += 1
    
    if n0 < 4:
        n0 = 4
    cdef np.ndarray[DTYPE_t,ndim=2] array_08 = array_0[:n0]
    #print(array_08)
    
    cdef float x0_x = np.mean(array_08.T[1])
    cdef float x0_y = np.mean(array_08.T[2])
    
    cdef float r_sigma
    for i in range(len(array_0)):
        if array_0[i][0] < 0.606:
            r_sigma = sqrt((array_0[i][1]-x0_x)**2+(array_0[i][2]-x0_y)**2)
            break
    if not 'r_sigma' in locals():
        r_sigma=3
    
    cdef list dist_array_l = []
    cdef float dist
    
    for i in range(n0):
        dist = sqrt((array_0[i][1]-x0_x)**2+(array_0[i][2]-x0_y)**2)
        dist_array_l.append(dist)
    
    cdef np.ndarray[DTYPE_t,ndim=1] dist_array = np.array(dist_array_l)            
    cdef np.ndarray[DTYPE_t,ndim=2] array_corners = np.zeros((4,3))
    
    #cdef np.ndarray[DTYPE_i,ndim=1] idx
    
    for i in range(4):
        idx = np.argmax(dist_array)
        dist_array[idx] = -1
        array_corners[i] = array_08[idx]

    #print(array_corners)
    cdef float max_dist = 0.0
    cdef signed int idx_1 = -1
    cdef signed int idx_2 = -1
    
    for i in range(4):
        for j in range(i):
            dist = sqrt((array_corners[i][1]-array_corners[j][1])**2+(array_corners[i][2]-array_corners[j][2])**2)
            if dist > max_dist:
                max_dist = dist
                idx_1 = i
                idx_2 = j
      
    cdef float x0_lx = max_dist
        
    cdef list idxs = [0,1,2,3]
    idxs.remove(idx_1)
    idxs.remove(idx_2)
    cdef float x0_ly = sqrt((array_corners[idxs[0]][1]-array_corners[idxs[1]][1])**2+(array_corners[idxs[0]][2]-array_corners[idxs[1]][2])**2)
    #x0_ly = x0_ly
    
    # get sigma and correct 
    cdef float r_l = (x0_lx+x0_ly)/2
    cdef float x0_sigma = min(max(1/0.332*(r_l-r_sigma),1),10)
    x0_lx = max(x0_lx - 0.668*x0_sigma,0)
    x0_ly = max(x0_ly - 0.668*x0_sigma,0)
    
    cdef list lb = [min(0.5*p_x0,3*p_x0), 1, 0, 0, 0, -90, min(array_08.T[1]), min(array_08.T[2])] #[x0[0], 0, 0, 0, -100, -90, min(array_n.T[1][nx[0]],array_n.T[1][nx[1]]), min(array_n.T[2][ny[0]],array_n.T[2][ny[1]])] #[x0[0], 0, 0, 0, 0.1, 0.1, 0, -20, 1, -90, min(array_n.T[1][nx[0]],array_n.T[1][nx[1]]), min(array_n.T[2][ny[0]],array_n.T[2][ny[1]])]
    cdef list ub = [max(0.5*p_x0,3*p_x0), 10, spacing*max(shape), spacing*max(shape), 0.05, 90,  max(array_08.T[1]), max(array_08.T[2])]
    
    
    cdef list x0_calc = [p_x0, x0_sigma, x0_lx, x0_ly, 0, 0, x0_x, x0_y]
    #cdef list x0_prev = list(x0_i)
    cdef list x0
    
    if np.all((list_x0.T[0] == 0)):
        x0 = x0_calc
        print('choose new')
    else:
        x0, lb, ub, list_x0 = find_best_x0(list_x0,lb,ub,spacing)
        if len(x0) == 0:
            x0 = x0_calc
            print('choose new')
        else:
            print('choose previous')
    
    #print(dict(zip(params_names,x0)))
    #lb = [2*array[-1][0], 0, 0, 0, 0, -90, spacing, spacing] #[x0[0], 0, 0, 0, -100, -90, min(array_n.T[1][nx[0]],array_n.T[1][nx[1]]), min(array_n.T[2][ny[0]],array_n.T[2][ny[1]])] #[x0[0], 0, 0, 0, 0.1, 0.1, 0, -20, 1, -90, min(array_n.T[1][nx[0]],array_n.T[1][nx[1]]), min(array_n.T[2][ny[0]],array_n.T[2][ny[1]])]
    #ub = [2*array[0][0], 10, spacing*max(shape), spacing*max(shape), 0.05, 90, spacing*(shape[0]), spacing*(shape[1])]
    
   
    
    cdef np.ndarray[DTYPE_t,ndim=1] ulb = np.array(ub[-2:])-np.array(lb[-2:])
    for i,b in enumerate(ulb):
        if b <= 0:
            lb[6+i] = lb[6+i]-0.5
            ub[6+i] = ub[6+i]+0.5
    
    #print(x0_calc)
    #print(x0_prev)
    #print(x0)
    #print(lb)
    #print(ub)
    return x0, lb, ub, list_x0
"""
def PressureDistribution(X,p0,std,lx,ly,r_curve,theta,x0,y0):
    #params = [p0,std,lx,ly,r_curve,F,theta,x0,y0]
    X_i = X[0]
    Y_i = X[1]
    Z_i = calc_Z(X_i,Y_i,p0,std,lx,ly,r_curve,theta,x0,y0)
    #plot_Z([5,9],4.5,X_new,Y_new,Z_new)
    #print("time AllInOne: "+str(t1-t0))
    #print("time getValue: "+str(t2-t1))
    #print(params)
    return Z_i

def PressureDistribution_cython(np.ndarray[DTYPE_t,ndim=2] X, float p0, float std, float lx, float ly, float r_curve1, float theta1, float x0, float y0):
    cdef float theta = theta/180*pi
    cdef unsigned int n = len(X[0])
    
    cdef np.ndarray[DTYPE_t] X_new,Y_new
    cdef float r_curve
    cdef unsigned int i
    cdef float x,y,dist
    
    cdef np.ndarray[DTYPE_t] X_theta = (X[0]-x0)*cos(theta)+(X[1]-y0)*sin(theta)
    cdef np.ndarray[DTYPE_t] Y_theta = -(X[0]-x0)*sin(theta)+(X[1]-y0)*cos(theta)
    
    
    if r_curve1 < -0.1:
        r_curve = -10/(10**(-r_curve-1))
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r + r_curve)
    elif r_curve1 > 0.1:
        r_curve = 10/(10**(r_curve-1))
        r = np.sqrt(X_theta**2+(Y_theta-r_curve)**2)
        alpha = np.arctan2(X_theta,-(Y_theta-r_curve))
        X_new = np.multiply(alpha,r)
        Y_new = (r - r_curve)
    else:
        X_new = X_theta.copy()
        Y_new = Y_theta.copy()
  
    for i in range(n):
        x = X_new[i]
        y = Y_new[i]
        
        if (fabs(x) < lx/2 and fabs(y) < ly/2):
            X[-1][i] = p0
        elif (fabs(x) > lx/2 and fabs(y) > ly/2):
            r = fmin((x-lx/2)**2+(y-ly/2)**2,
                fmin((x+lx/2)**2+(y-ly/2)**2,
                fmin((x-lx/2)**2+(y+ly/2)**2,
                (x+lx/2)**2+(y+ly/2)**2)))
            r = sqrt(r)
            X[-1][i] = p0*exp(-0.5*(r/std)**2)
        elif x < -lx/2:
            r = -lx/2-x
            X[-1][i] = p0*exp(-0.5*(r/std)**2)
        elif x > lx/2:
            r = x-lx/2
            X[-1][i] = p0*exp(-0.5*(r/std)**2)
        elif y < -ly/2:
            r = -ly/2-y
            X[-1][i] = p0*exp(-0.5*(r/std)**2)
        elif y > ly/2:
            r = y-ly/2
            X[-1][i] = p0*exp(-0.5*(r/std)**2)
        else:
            X[-1][i] = p0

    return X[-1]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[DTYPE_t,ndim=1] PressureDistribution_cython_parallel(np.ndarray[DTYPE_t,ndim=2] X, float p0, float std, float lx, float ly, float r_curve1, float theta1, float x0, float y0):
    cdef float theta = theta1/180*pi
    cdef Py_ssize_t n = len(X[0])
    
    #cdef np.ndarray[DTYPE_t] X_new,Y_new
    cdef float r_curve
    cdef signed int i
    cdef float x,y,dist
    cdef float x_theta, y_theta, r, alpha
    
    if r_curve1 < -0.1:
        r_curve = -10/(10**(-r_curve1-1))
        with nogil, parallel():
            for i in prange(n,schedule='static'):
                x_theta = (X[0,i]-x0)*cos(theta)+(X[1,i]-y0)*sin(theta)
                y_theta = -(X[0,i]-x0)*sin(theta)+(X[1,i]-y0)*cos(theta)
                r = sqrt(x_theta**2+(y_theta-r_curve)**2)
                alpha = atan2(x_theta,y_theta-r_curve)
                x = alpha*r
                y = r+r_curve
                if (fabs(x) < lx/2 and fabs(y) < ly/2):
                    X[2,i] = p0
                elif (fabs(x) > lx/2 and fabs(y) > ly/2):
                    r = fmin((x-lx/2)**2+(y-ly/2)**2,
                        fmin((x+lx/2)**2+(y-ly/2)**2,
                        fmin((x-lx/2)**2+(y+ly/2)**2,
                        (x+lx/2)**2+(y+ly/2)**2)))
                    r = sqrt(r)
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif x < -lx/2:
                    r = -lx/2-x
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif x > lx/2:
                    r = x-lx/2
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif y < -ly/2:
                    r = -ly/2-y
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif y > ly/2:
                    r = y-ly/2
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                else:
                    X[2,i] = p0
    elif r_curve1 > 0.1:
        r_curve = 10/(10**(r_curve1-1))
        with nogil, parallel():
            for i in prange(n,schedule='static'):
                x_theta = (X[0,i]-x0)*cos(theta)+(X[1,i]-y0)*sin(theta)
                y_theta = -(X[0,i]-x0)*sin(theta)+(X[1,i]-y0)*cos(theta)
                r = sqrt(x_theta**2+(y_theta-r_curve)**2)
                alpha = atan2(x_theta,-(y_theta-r_curve))
                x = alpha*r
                y = r - r_curve
                if (fabs(x) < lx/2 and fabs(y) < ly/2):
                    X[2,i] = p0
                elif (fabs(x) > lx/2 and fabs(y) > ly/2):
                    r = fmin((x-lx/2)**2+(y-ly/2)**2,
                        fmin((x+lx/2)**2+(y-ly/2)**2,
                        fmin((x-lx/2)**2+(y+ly/2)**2,
                        (x+lx/2)**2+(y+ly/2)**2)))
                    r = sqrt(r)
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif x < -lx/2:
                    r = -lx/2-x
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif x > lx/2:
                    r = x-lx/2
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif y < -ly/2:
                    r = -ly/2-y
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif y > ly/2:
                    r = y-ly/2
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                else:
                    X[2,i] = p0
    else:
        with nogil, parallel():
            for i in prange(n,schedule='static'):
                x = (X[0,i]-x0)*cos(theta)+(X[1,i]-y0)*sin(theta)
                y = -(X[0,i]-x0)*sin(theta)+(X[1,i]-y0)*cos(theta)
                if (fabs(x) < lx/2 and fabs(y) < ly/2):
                    X[2,i] = p0
                elif (fabs(x) > lx/2 and fabs(y) > ly/2):
                    r = fmin((x-lx/2)**2+(y-ly/2)**2,
                        fmin((x+lx/2)**2+(y-ly/2)**2,
                        fmin((x-lx/2)**2+(y+ly/2)**2,
                        (x+lx/2)**2+(y+ly/2)**2)))
                    r = sqrt(r)
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif x < -lx/2:
                    r = -lx/2-x
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif x > lx/2:
                    r = x-lx/2
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif y < -ly/2:
                    r = -ly/2-y
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                elif y > ly/2:
                    r = y-ly/2
                    X[2,i] = p0*exp(-0.5*(r/std)**2)
                else:
                    X[2,i] = p0

    return X[2]
"""  
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[DTYPE_t,ndim=1] PressureDistribution_cython(np.ndarray[DTYPE_t,ndim=2] X, float p0, float std, float lx, float ly, float r_curve1, float theta1, float x0, float y0):
    cdef float theta = theta1/180*pi
    cdef Py_ssize_t n = len(X[0])
    
    #cdef np.ndarray[DTYPE_t] X_new,Y_new
    cdef float r_curve
    cdef signed int i
    cdef float x,y,dist
    cdef float x_theta, y_theta, r, alpha
    
    if r_curve1 < -0.1:
        r_curve = -10/(10**(-r_curve1-1))
        for i in range(n):
            x_theta = (X[0,i]-x0)*cos(theta)+(X[1,i]-y0)*sin(theta)
            y_theta = -(X[0,i]-x0)*sin(theta)+(X[1,i]-y0)*cos(theta)
            r = sqrt(x_theta**2+(y_theta-r_curve)**2)
            alpha = atan2(x_theta,y_theta-r_curve)
            x = alpha*r
            y = r+r_curve
            if (fabs(x) < lx/2 and fabs(y) < ly/2):
                X[2,i] = p0
            elif (fabs(x) > lx/2 and fabs(y) > ly/2):
                r = fmin((x-lx/2)**2+(y-ly/2)**2,
                    fmin((x+lx/2)**2+(y-ly/2)**2,
                    fmin((x-lx/2)**2+(y+ly/2)**2,
                    (x+lx/2)**2+(y+ly/2)**2)))
                r = sqrt(r)
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif x < -lx/2:
                r = -lx/2-x
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif x > lx/2:
                r = x-lx/2
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif y < -ly/2:
                r = -ly/2-y
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif y > ly/2:
                r = y-ly/2
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            else:
                X[2,i] = p0
    elif r_curve1 > 0.1:
        r_curve = 10/(10**(r_curve1-1))
        for i in range(n):
            x_theta = (X[0,i]-x0)*cos(theta)+(X[1,i]-y0)*sin(theta)
            y_theta = -(X[0,i]-x0)*sin(theta)+(X[1,i]-y0)*cos(theta)
            r = sqrt(x_theta**2+(y_theta-r_curve)**2)
            alpha = atan2(x_theta,-(y_theta-r_curve))
            x = alpha*r
            y = r - r_curve
            if (fabs(x) < lx/2 and fabs(y) < ly/2):
                X[2,i] = p0
            elif (fabs(x) > lx/2 and fabs(y) > ly/2):
                r = fmin((x-lx/2)**2+(y-ly/2)**2,
                    fmin((x+lx/2)**2+(y-ly/2)**2,
                    fmin((x-lx/2)**2+(y+ly/2)**2,
                    (x+lx/2)**2+(y+ly/2)**2)))
                r = sqrt(r)
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif x < -lx/2:
                r = -lx/2-x
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif x > lx/2:
                r = x-lx/2
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif y < -ly/2:
                r = -ly/2-y
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif y > ly/2:
                r = y-ly/2
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            else:
                X[2,i] = p0
    else:
        for i in range(n):
            x = (X[0,i]-x0)*cos(theta)+(X[1,i]-y0)*sin(theta)
            y = -(X[0,i]-x0)*sin(theta)+(X[1,i]-y0)*cos(theta)
            if (fabs(x) < lx/2 and fabs(y) < ly/2):
                X[2,i] = p0
            elif (fabs(x) > lx/2 and fabs(y) > ly/2):
                r = fmin((x-lx/2)**2+(y-ly/2)**2,
                    fmin((x+lx/2)**2+(y-ly/2)**2,
                    fmin((x-lx/2)**2+(y+ly/2)**2,
                    (x+lx/2)**2+(y+ly/2)**2)))
                r = sqrt(r)
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif x < -lx/2:
                r = -lx/2-x
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif x > lx/2:
                r = x-lx/2
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif y < -ly/2:
                r = -ly/2-y
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            elif y > ly/2:
                r = y-ly/2
                X[2,i] = p0*exp(-0.5*(r/std)**2)
            else:
                X[2,i] = p0

    return X[2]

def Optimization(np.ndarray[DTYPE_t,ndim=1] xi, np.ndarray[DTYPE_t,ndim=1] yi, np.ndarray[DTYPE_t,ndim=1] pressureArray, list shape, float spacing,  np.ndarray[DTYPE_t,ndim=2] array_rbf, np.ndarray[DTYPE_t,ndim=2] list_x0,int n=9, int it_max = 10, int t_max = 40):
    """
    n: amount of points used
    """
    #global x0
    cdef unsigned int n_extra = n
    cdef unsigned int it = 0
    cdef list list_E = []
    
    cdef np.ndarray[DTYPE_l,ndim=1] order_rbf = np.argsort(array_rbf[2])
    order_rbf = np.flip(order_rbf)
    array_rbf[0] = array_rbf[0][order_rbf]
    array_rbf[1] = array_rbf[1][order_rbf]
    array_rbf[2] = array_rbf[2][order_rbf]
    # t0 = time.time()
    cdef unsigned int amount = int(shape[0]*shape[1])
    cdef unsigned int n_all = amount+n_extra
    cdef np.ndarray[DTYPE_t,ndim=1] Zi = np.zeros((amount+n_extra,))
    cdef np.ndarray[DTYPE_t,ndim=2] array = np.zeros((amount+n_extra,3)) # n points with highest pressure; 0: pressure; 1: x coordinate; 2: y coordinate
        
    array.T[0] = np.append(pressureArray,array_rbf[2][:n_extra])
    array.T[1] = np.append(xi,array_rbf[0][:n_extra])
    array.T[2] = np.append(yi,array_rbf[1][:n_extra])
    
    array.T[0] = array.T[0]- Zi  
    cdef float time_opt = 0
    
    cdef list x_scale
    cdef np.ndarray[DTYPE_t,ndim=2] xdata
    cdef np.ndarray[DTYPE_t,ndim=1] ydata
    cdef list meth = ['trf','dogbox']
    
    cdef float t_est = 0.0
    cdef double to = time.time()
    
    while (max(abs(array.T[0])/max(pressureArray)) > 0.1) and (it < it_max) and (time_opt+t_est < t_max):
        #array = np.zeros((amount,3)) 
        order = np.argsort(array.T[0])
        order = np.flip(order)
        
        array.T[0] = array.T[0][order]
        array.T[1] = array.T[1][order]
        array.T[2] = array.T[2][order]
        #print("p_error_max: "+str(max(abs(array.T[0])/max(pressureArray))))
        #print(array)
        """
        for i in range(amount):
            idx = np.argmax(pressureArray)
            array[i][0] = pressureArray[idx]    
            array[i][1] = xi[idx]
            array[i][2] = yi[idx]
            
            pressureArray[idx] = -10**9
        """
        
        x0, lb, ub, list_x0 = goodGuess(array[:(n_all)],shape,spacing,list_x0)
        
        x_scale = [abs(x0[0]),5,spacing*max(shape), spacing*max(shape),3,90,spacing*(max(shape)),spacing*(max(shape))]
        #time0 = time.time()
        
        xdata = array.T[1:].T[:n_all].T
        xdata = np.append(xdata,np.zeros((1,xdata.shape[1])),axis=0)
        ydata = array.T[0][:n_all]
        
        #print(xdata)
        #print(ydata)
        
        E = curve_fit(PressureDistribution_cython, xdata, ydata, x0, bounds=(lb, ub), check_finite = False, method = meth[1], maxfev=100,verbose=0,ftol=1e-4,diff_step=0.15,x_scale=x_scale)#,diff_step=100)
        #print(E.x)
        list_E.append(E)
       
        Zi = calc_Z(array.T[1], array.T[2], E.x[0], E.x[1], E.x[2], E.x[3], E.x[4], E.x[5], E.x[6], E.x[7])
        array.T[0] = array.T[0]- Zi
        it += 1
        time_opt += (time.time()-to)*10**3
        to = time.time()
        t_est = time_opt/it
        
        #print(E)
        ### denormalize
        #E.x[0] = E.x[0]*np.max(array.T[1])
        #E.x[4] = E.x[4]*length_n
        #E.x[5] = E.x[5]*length_n
        """
        if E.success:
            x0 = E.x
        """
        #E.x[0] = E.x[0]/(E.x[1])**2 #==> E.x = [p0,z,x_c,y_c] p0 in Pa
        #print(E)
        #print(1/(time.time()-t0))
        
    #array = np.zeros((amount,3)) # n points with highest pressure; 2: pressure; 3: x coordinate; 4: y coordinate
        
    order = np.argsort(array.T[0])
    order = np.flip(order)
    
    array.T[0] = array.T[0][order]
    array.T[1] = array.T[1][order]
    array.T[2] = array.T[2][order]
    #print("p_error_max: "+str(max(abs(array.T[0])/max(pressureArray))))
    #print(array)
    
        
    if (it + 1 == it_max):
        print("maximum number of iteration ("+str(it_max)+") reached")
    elif (time_opt+t_est > t_max):
        print("time limit reached. "+str(it)+" iteration done")
    else:
        print("solution has reached goal accuracy after "+str(it)+" iterations")
        
    #print("time in curve_fit: "+str(time_opt))
    return array,list_E
