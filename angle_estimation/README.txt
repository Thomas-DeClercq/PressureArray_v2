In data, pos angle = counter clockwise rotation


STEP4_2_90, STEP4_10_90 and STEP4_40_90: angle = [0,90]
#################################################################################
Data1: 1 timestep, fixed position (middle), fixed pressure distribution, it = 1, only rotational slip
Data2: multiple timesteps (10), fixed position (middle), fixed pressure distribution, it = 1, only rotational slip
Data3: multiple timesteps (10), variable position and orientation, fixed pressure distribution, it = 1, only rotational slip
Data4: multiple timesteps (10), variable position and orientation, variable pressure distribution, it = 1, only rotational slip
Data5: multiple timesteps (10), variable position and orientation, variable pressure distribution, it = 5, only rotational slip
Data6: 1 timesteps, variable position and orientation, variable pressure distribution, it = 5, only rotational slip
Data7: 1 timesteps, variable position and orientation, variable pressure distribution (with restrictions7), it = 5, only rotational slip
Data8: 1 timesteps, variable position and orientation, variable pressure distribution (with restrictions8), it = 5, only rotational slip
Data9: 1 timesteps, variable position and orientation, variable pressure distribution (with restrictions9), it = 5, only rotational slip

Data10: 10 timesteps,variable position and orientation, variable pressure distribution (no restrictions), it = 1-5, only rotational slip
Data11: 10 timesteps,variable position and orientation, variable pressure distribution (|r| < 1), it = 1-5, only rotational slip

################################################################################

Restrictions7:
it = 1
slip angle = [-45,45]
ly = [0,0.5*lx]
r = 0
--> average error: 4°, median error: 0.7° (with r=0 in reconstruction)
--> average error: 7°, median error: 1.3°
--> PCA_prop_error: 11°, PCA_exp_error: 10° (med: 5.2°)


Restrictions7_1:
it = 1
slip angle = [-90,90]
ly = [0,lx]
r = 0

Restrictions8:
it = [1,3,5]
slip angle = [-45,45]
ly = [0,0.5*lx]
r = 0
--> average error: 9°, median error: 2.7° (with r=0 in reconstruction)
--> average error: 11°, median error: 6.1°
--> PCA_prop_error: 12°, PCA_exp_error: 9.2° (med: 4.9°)

Restrictions8_1:
it = [1,2,3,4,5]
slip angle = [-45,45]
ly = [0,lx]
r = 0
--> average error: 11°, median error: 3.9° (with r=0 in reconstruction)
--> average error: 13°, median error: 8.1°
--> PCA_prop_error: 14°, PCA_exp_error: 12° (med: 6.9°)

Restrictions9:
it = 1
slip angle = [-45,45]
ly = [0,0.5*lx]
r = free
--> average error: 17°, median error: 12°
--> PCA_prop_error: 17°, PCA_exp_error: 19° (med: 12.5°)

Data10 & Data11:
it = 1-5
slip angle = [-180,180]

