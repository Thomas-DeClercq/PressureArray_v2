import numpy as np

import os

all_names = os.listdir('./hyperparameters_optimization/error_arrays')

all_errors = np.array([])
list_i = []

for name in all_names:
    error_array_i = np.load(f'./hyperparameters_optimization/error_arrays/{name}')
    i = int(name.split('_')[2])
    if True:
        #print(name)
        aver = np.average(error_array_i[-1])
        med = np.median(error_array_i[-1])
        std = np.std(error_array_i[-1])
        list_i.append(i)

        error_i = np.array([i,aver,med,std])
        try:
            all_errors = np.vstack((all_errors,error_i))
        except:
            all_errors = np.append(all_errors,error_i)


all_aver = all_errors.T[1]
all_med = all_errors.T[2]
all_std = all_errors.T[3]


print(f'average average_error: {np.average(all_aver)}')
print(f'min average error: {np.min(all_aver)} @ {list_i[np.argmin(all_aver)]}')

print(f'average median_error: {np.average(all_med)}')
print(f'min median error: {np.min(all_med)} @ {list_i[np.argmin(all_med)]}')

print(f'average std_error: {np.average(all_std)}')
print(f'std @ min average error: {all_std[np.argmin(all_aver)]}')
print(f'std @ min median error: {all_std[np.argmin(all_med)]}')

