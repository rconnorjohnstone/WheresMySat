# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 23:25:12 2019

@author: chels
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:55:27 2019

@author: chels
"""
import numpy as np
from measurements import stationloc, stationstate, genel, genmeas, genHtilde
from coe2rv import rad2deg, coe2rvfunc, deg2rad
from scipy.integrate import solve_ivp
from EOMs_J2_n_6 import eom


class measurementclss:
    def __init__(self,meas,time,stationID, H_tilde, r, sigma):
        self.meas = meas
        self.time = time
        self.stationID = stationID
        self.H_tilde = H_tilde
        self.r = r
        self.sigma = sigma

#dummy variables
H = np.ones((1,6))
r = np.full(1,1e-8)
sigma = 1e-8

pert = np.array([.001, 0, 0, 0., .000001, 0.])
def genmeas2(inttime, timestep, y0, save):
    theta0 = 122.
    we = 2 * np.pi / (24.* 3600.)
    zenelmask = np.pi / 2 - 10.* np.pi / 180.
    stations = stationloc()
    measurements = []
    for j in range(int(np.floor(inttime / timestep))):
        if j == 0:
            state_sc = y0
        #calculate s/c position
        state_sol = solve_ivp(eom,[10.*j, 10.*(j+1)] ,state_sc, max_step=5)
        state_sc = state_sol.y[:,-1] 
        for i in range(len(stations)):
            state_s = stationstate(stations[i][0], stations[i][1], theta0,10.*(j+1), we)
            H = genHtilde(state_sc, state_s)
            H = H[1,:].reshape((1,6))
            zenel = genel(state_sc, state_s)
            if zenel < zenelmask:
                rho, rhod = genmeas(state_sc, state_s)
                #append [time stamp, station #, elevation, range, range rate]
                measurements.append(measurementclss(rhod, state_sol.t[-1], \
                                                    i+1,H, r, sigma))
    measurements = np.array(measurements)
    if save == 1:
#        np.save('scstates',scstates)
        np.save('meas_nonoise', measurements)
#        np.save('stationstates', stationstates)
    return measurements

def genMSM():
    x0 = list(coe2rvfunc(10000.,.001,deg2rad(40.),deg2rad(80.),deg2rad(40.),0.).reshape(6))
    msm = genmeas2(5000,10,np.array(x0),0)
    
#    sort by stationID
#    msm.tolist().sort(key = lambda x: x.stationID)
    #return all times
#    [msr.stationID for msr in msm]
    return msm