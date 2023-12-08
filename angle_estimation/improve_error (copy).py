#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:35:22 2023

@author: pc-robotiq
"""
import os
import pandas as pd
import numpy as np

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib import cm, animation
import matplotlib.patches as patches
from math import pi
import random
import time

import scipy
from scipy.interpolate import RBFInterpolator, Rbf

from PressureReconstruction import calc_Z, Optimization

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pickle
#pickle.dump(svr,open('./numpy_arrays/svr_STEP4_40_180_02i.sav','wb'))

#from generatePressureDistributionsInTime import animate

if __name__ == "__main__":
    error_array_rel = np.load('./numpy_arrays/error_array_STEP4_180_abs.npy')
    timesteps = 9
    print(f"average: {np.average(error_array_rel[2])}")
    print(f"median: {np.median(error_array_rel[2])}")
    print(f"std: {np.std(error_array_rel[2])}")
    for i in range(20):
        fig1 = plt.figure()
        ax = plt.subplot(111)
        ax.plot(error_array_rel[0][i*timesteps:(i+1)*timesteps-1],label='real angle')
        ax.plot(error_array_rel[1][i*timesteps:(i+1)*timesteps-1],label='estimated angle')
        ax.legend()
        plt.show()