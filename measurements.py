# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 18:16:04 2019

@author: chels
"""
import numpy as np
from EOMs_J2_n_6 import eom
from EOMs import eom_wJ3
from scipy.integrate import solve_ivp
from coe2rv import coe2rvfunc, rad2deg, deg2rad
from EOMs_J2_n_6 import keplerJ2_wPhi_ODE_thangavelu, constants


#Generate ground truth measurements and noisy measurements

#input [latitude(deg),longitude(deg),initial ang. b/w ECI and ECEF(deg),time elapsed, 
#earth rotation rate]
def stationstate(phi, lambd, theta0, t, we):
    phi = 90. - phi
    phi *= np.pi / 180.
    lambd *= np.pi / 180.
    theta0 *= np.pi / 180.
    R_E = 6378.
    x = R_E * np.sin(phi) * np.cos(lambd + theta0 + we * t)
    y = R_E * np.sin(phi) * np.sin(lambd + theta0 + we * t)
    z = R_E * np.cos(phi)
    xd = - R_E * np.sin(phi) * np.sin(lambd + theta0 + we * t) * we
    yd = R_E * np.sin(phi) * np.cos(lambd + theta0 + we * t) * we
    zd = 0.
    return np.array([x, y, z, xd, yd, zd])

def stationstate2(stationID, t):
    i = stationID - 1
    we = 2. * np.pi / (24.* 3600.)
    stations = np.array([[-35.398333, 148.981944],[40.427222, 355.749444 ], \
                         [35.247164, 243.205]])
    theta0 = 122.
    state_stn = stationstate(stations[i][0], stations[i][1], theta0, t, we)
    return state_stn

def stationloc():
    return np.array([[-35.398333, 148.981944], [40.427222, 355.749444 ], \
                     [35.247164, 243.205]])
    
def genel(state_sc,state_s):
    zenel = np.arccos(np.dot(state_sc[0:3], state_s[0:3]) / (np.linalg.norm(state_sc[0:3]) * np.linalg.norm(state_s[0:3])))
    los = state_sc[0:3] - state_s[0:3]
    zenel = np.arccos(np.dot(state_s[0:3], los) / (np.linalg.norm(state_s[0:3]) * np.linalg.norm(los)))
    return zenel

def genmeas(state_sc, state_s):
    rho = np.linalg.norm(state_sc[0:3] - state_s[0:3])
    rhod = np.dot((state_sc[0:3] - state_s[0:3]),(state_sc[3:] - state_s[3:]))/rho
    return rho, rhod

def G(state_star, ti, stationID):
    state_s = stationstate2(stationID, ti)
    rho, rhod = genmeas(state_star, state_s)
    return rho, rhod

def genmeas2(inttime, timestep, y0, save):
    theta0 = 122.
    we = 2 * np.pi / (24.* 3600.)
    zenelmask = np.pi / 2 - 10.* np.pi / 180.
    stations = stationloc()
    measurements = []
    stationstates = []
    scstates = []
    for j in range(int(np.floor(inttime / timestep))):
        if j == 0:
            state_sc = y0
        #calculate s/c position
        state_sol = solve_ivp(eom,[10.*j, 10.*(j+1)] ,state_sc, max_step=5)
        state_sc = state_sol.y[:,-1] 
        scstates.append([state_sol.t[-1], state_sc[0], state_sc[1], state_sc[2], \
                        state_sc[3], state_sc[4], state_sc[5]])
        for i in range(len(stations)):
            state_s = stationstate(stations[i][0], stations[i][1], theta0,10.*(j+1), we)
            zenel = genel(state_sc, state_s)
            if zenel < zenelmask:
                rho,rhod = genmeas(state_sc, state_s)
                el = rad2deg(np.pi/2 - zenel)
                #append [time stamp, station #, elevation, range, range rate]
                measurements.append([state_sol.t[-1], i+1, el, rho, rhod])
                stationstates.append([state_sol.t[-1], i+1, state_s[0], state_s[1], \
                                      state_s[2], state_s[3], state_s[4],state_s[5]])
    scstates = np.array(scstates)
    stationstates = np.array(stationstates)
    measurements = np.array(measurements)
    if save == 1:
#        np.save('scstates',scstates)
        np.save('meas_nonoise', measurements)
        np.save('stationstates', stationstates)
    return measurements


    

def genmeas2_wJ3(inttime, timestep, y0,save):
    theta0 = 122.
    we = 2 * np.pi/ (24.* 3600.)
    zenelmask = np.pi / 2 - 10. * np.pi / 180.
    stations = stationloc()
    measurements = []
    stationstates = []
    scstates = []
    for j in range(int(np.floor(inttime / timestep))):
        if j == 0:
            state_sc = y0
        #calculate s/c position
        state_sol = solve_ivp(eom_wJ3, [10.*j,10.*(j+1)], state_sc, max_step=5)
        state_sc = state_sol.y[:,-1]
        scstates.append([state_sol.t[-1], state_sc[0], state_sc[1], state_sc[2], \
                        state_sc[3], state_sc[4], state_sc[5]])
        for i in range(len(stations)):
            state_s = stationstate(stations[i][0], stations[i][1], theta0, 10.*(j+1), we)
            zenel = genel(state_sc, state_s)
            if zenel < zenelmask:
                rho, rhod = genmeas(state_sc, state_s)
                el = rad2deg(np.pi / 2 - zenel)
                #append [time stamp, station #, elevation, range, range rate]
                measurements.append([state_sol.t[-1], i+1, el, rho, rhod])
                stationstates.append([state_sol.t[-1], i+1, state_s[0], state_s[1],\
                                      state_s[2], state_s[3], state_s[4], state_s[5]])
    scstates = np.array(scstates)
    stationstates = np.array(stationstates)
    measurements = np.array(measurements)
    if save == 1:
#        np.save('scstates',scstates)
        np.save('meas_nonoise', measurements)
        np.save('stationstates', stationstates)
    return measurements

def addnoise(meas_groundtruth, rhonoise, rhodnoise, save):
#    add noise to data from data log (only run once then call from data file)
    rhonoise = 1.e-3
    rhodnoise = 1.e-6
    measurements = np.array([meas_groundtruth[:,0], meas_groundtruth[:,1], meas_groundtruth[:,2],\
                            meas_groundtruth[:,3] + np.random.normal(0., rhonoise, (np.shape(meas_groundtruth)[0])), \
                            meas_groundtruth[:,4] + np.random.normal(0., rhodnoise, (np.shape(meas_groundtruth)[0]))]).T
    if save==1:
        np.save('noisymeas', measurements)
    return measurements

def genHtilde(x_sc,x_obs):
    #unpack spacecraft state
    Rs_x = x_sc[0]
    Rs_y = x_sc[1]
    Rs_z = x_sc[2]
    Vs_x = x_sc[3]
    Vs_y = x_sc[4]
    Vs_z = x_sc[5]
    #unpack observer state
    R_x = x_obs[0]
    R_y = x_obs[1]
    R_z = x_obs[2]
    V_x = x_obs[3]
    V_y = x_obs[4]
    V_z = x_obs[5]
    #calculate range and range rate
    rho = np.sqrt((R_x - Rs_x)**2+(R_y - Rs_y)**2+(R_z - Rs_z)**2)
#    rhod = ((R_x - Rs_x)*(V_x - Vs_x) + (R_y - Rs_y) * (V_y - Vs_y) + \
#            (R_z - Rs_z) * (V_z - Vs_z)) / rho
    #calculate range partials
    dh1drx = (Rs_x - R_x) / rho
    dh1dry = (Rs_y - R_y) / rho
    dh1drz = (Rs_z - R_z) / rho
    dh1dvx = 0.
    dh1dvy = 0.
    dh1dvz = 0.
    #calculate range rate partials
    dh2drx = ((2 * R_x - 2 * Rs_x) * ((R_x - Rs_x) * (V_x - Vs_x) + (R_y - Rs_y) * (V_y - Vs_y)  \
        + (R_z - Rs_z) * (V_z - Vs_z))) / (2 * ((R_x - Rs_x)**2 + (R_y - Rs_y)**2 +  \
        (R_z - Rs_z)**2)**(3/2)) - (V_x - Vs_x) / ((R_x - Rs_x)**2 + (R_y - Rs_y)**2 +  \
        (R_z - Rs_z)**2)**(1/2)
    dh2dry = ((2 * R_y - 2*Rs_y) * ((R_x - Rs_x) * (V_x - Vs_x) + (R_y - Rs_y) * (V_y - Vs_y) + (R_z - Rs_z) * (V_z - Vs_z))) / (2 * ((R_x - Rs_x)**2 + (R_y - Rs_y)**2 + (R_z - Rs_z)**2)**(3/2)) - (V_y - Vs_y)/((R_x - Rs_x)**2 + (R_y - Rs_y)**2 + (R_z - Rs_z)**2)**(1/2)
    dh2drz = (( 2 *R_z - 2*Rs_z) * ((R_x - Rs_x)*(V_x - Vs_x) + (R_y - Rs_y) * (V_y - Vs_y) + (R_z - Rs_z) * (V_z - Vs_z))) / (2 * ((R_x - Rs_x)**2 + (R_y - Rs_y)**2 + (R_z - Rs_z)**2)**(3/2)) - (V_z - Vs_z)/((R_x - Rs_x)**2 + (R_y - Rs_y)**2 + (R_z - Rs_z)**2)**(1/2)
    dh2dvx = - (R_x - Rs_x)/((R_x - Rs_x)**2 + (R_y - Rs_y)**2 + (R_z - Rs_z)**2)**(1/2)
    dh2dvy = - (R_y - Rs_y)/((R_x - Rs_x)**2 + (R_y - Rs_y)**2 + (R_z - Rs_z)**2)**(1/2)
    dh2dvz = - (R_z - Rs_z)/((R_x - Rs_x)**2 + (R_y - Rs_y)**2 + (R_z - Rs_z)**2)**(1/2)
    Htilde_sc_obs = np.array([[dh1drx, dh1dry, dh1drz, dh1dvx, dh1dvy, dh1dvz], \
        [dh2drx, dh2dry, dh2drz, dh2dvx, dh2dvy, dh2dvz]])
    return Htilde_sc_obs

def findgaps(measurements):
    gaps = []
    for i in range(np.shape(measurements)[0]):
        t = 10 * (i+1)
        if t not in measurements[:,0]:
            gaps.append(t)
    return gaps

def rms(data):
    rmss = np.linalg.norm(data - np.mean(data))
    return rmss

#create and save trajectories:
def init_IC():
    x0 = list(coe2rvfunc(10000.,.001,deg2rad(40.),deg2rad(80.),deg2rad(40.),0.).reshape(6))
    a = 10000#km
    mu = 3.986004415e5 #km^3/s^2
    T = 2.*np.pi/np.sqrt(mu/a**3)
    dx0 = np.array([.001, 0., 0., 0., .00005**2, 0.])
    return x0, T, dx0
    
def init_meas():
    x0, T, dx0 = init_IC()
    meas_nonoise = genmeas2(T*15,10,np.array(x0)+dx0,1)
    rhonoise = 1.e-3
    rhodnoise = 1.e-6
    addnoise(meas_nonoise,rhonoise,rhodnoise,1)
    return None
    
def init_expmeas():
    x0, T, dx0 = init_IC()
#    dx0 = np.array([.1,0.,0.,0.,.00005**2,0.])
    meas_expected = genmeas2(T*15,10,np.array(x0),0)
#    meas_expected = genmeas2(T*15,10,np.array(x0)+dx0,0)
    np.save('meas_expected',meas_expected)
    return None

def genR():
    R = np.diagflat([1e-3**2,1e-6**2])
    return R

def init_refx0():
    x0 = init_IC()[0]
    phi0 = np.eye(6)
    dx0 = np.array([0., 0., 0., 0., 0., 0.]) #for testing
#    dx0 = np.array([.1,0.,0.,0.,.00005**2,0.])
    X_star0 = np.concatenate((np.array(x0)+dx0,phi0.flatten()))
    return X_star0

def init_traj_unpert():
    noisymeas = np.load('noisymeas.npy')
    x0, T, dx0 = init_IC()
    ts = np.concatenate((np.array([0]),noisymeas[:,0]))
    traj_unpert = solve_ivp(keplerJ2_wPhi_ODE_thangavelu,[0,ts[-1]],\
                        np.concatenate((np.array(x0),np.eye(6).flatten())),\
                        t_eval = ts,max_step=5)
    np.save('traj_unperty',traj_unpert.y)
    np.save('traj_unpertt',traj_unpert.t)
    return None

def init_traj_pert():
    noisymeas = np.load('noisymeas.npy')
    x0, T, dx0 = init_IC()
    ts = np.concatenate((np.array([0]),noisymeas[:,0]))
    traj_pert = solve_ivp(keplerJ2_wPhi_ODE_thangavelu,[0,ts[-1]],\
                        np.concatenate((np.array(x0)+dx0,np.eye(6).flatten())),\
                        t_eval = ts,max_step=5)
    np.save('traj_perty',traj_pert.y)
    np.save('traj_pertt',traj_pert.t)
    return None


def init_all():
    init_meas()
    init_expmeas()
    init_refx0()
    init_traj_unpert()
    return None

def init_loadfiles():
    x0, T, dx0 = init_IC()
    meas_nonoise = np.load('meas_nonoise.npy') #no initial pert
    noisymeas = np.load('noisymeas.npy') #no initial pert w noise
    meas_expected = np.load('meas_expected.npy') #initial pert
    R = genR()
    X_star0 = init_refx0()
    traj_unperty = np.load('traj_unperty.npy')
    traj_unpertt = np.load('traj_unpertt.npy')
    traj_perty = np.load('traj_perty.npy')
    traj_pertt = np.load('traj_pertt.npy')
    initdict = {'x0': x0, 'dx0': dx0, 'T': T, 'meas_nonoise': meas_nonoise, \
                'noisymeas': noisymeas, 'meas_expected': meas_expected, \
                'R': R, 'X_star0': X_star0, 'traj_unperty': traj_unperty, \
                'traj_unpertt': traj_unpertt, 'traj_perty': traj_perty, \
                'traj_pertt': traj_pertt}
    return initdict
    