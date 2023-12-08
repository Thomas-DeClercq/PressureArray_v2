import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from std_msgs.msg import Bool

import numpy as np
import scipy
import time
import pickle

import _thread as thread

it_max = 1
params = np.zeros((it_max,8))
params[0][1] = 1
params_ref = np.zeros((it_max,8))
params_ref[0][1] = 1

def readTopic(data):
    global params
    rows = int(len(data.data)/8)
    #params = np.zeros((rows,8))

    for i in range(rows):
        for j in range(8):
            params[i][j] = data.data[i*8+j]
    #print(params)
    return

def set_ref_parameters():
    global params_ref
    params_ref = params
    return

def reset_ref_parameters():
    global params_ref
    params_ref = np.zeros((1,8))
    params_ref[0][1] = 1
    return

def manual_recalibration():
    while not rospy.is_shutdown():
        text = input("Press 0 to recalibrate pressure sensors\n")
        if text == '0':
            set_ref_parameters()
        else:
            print("input is not 0")
        time.sleep(0.1)
    return

if __name__ == '__main__':
    rospy.init_node('RTAngleEstimation')
    freq = 20
    rate = rospy.Rate(freq)
    rospy.Subscriber('/estimatedParameters',Float64MultiArray,readTopic,queue_size=1)

    svr_name = 'Data3_1'
    svr = pickle.load(open(f'./Data_RT/SVR/{svr_name}.sav','rb'))

    thread.start_new_thread(manual_recalibration,())

    while not rospy.is_shutdown():
        if params[0][0] > 1000:
            if np.sum(params_ref) == 1:
                set_ref_parameters()
            
            params_est = np.append(params_ref.flatten(),params.flatten())
            params_est = params_est.reshape(1,-1) #single sample
            angle_est = svr.predict(params_est)
            print(angle_est)
        else:
            reset_ref_parameters()
        rate.sleep()
