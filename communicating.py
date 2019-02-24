import numpy as np
from numpy import linalg as LA
import requests
import WheresMySat as wms
import time
from WheresMySat import RRE_measurements as RRE
import json
import random

# -----------------------------------------------------------------------------------------------

def oe_to_rth(oe,mu):
    ''' Converts a 6 element list or ndarray of the form: semi-major axis, eccentricity,
     inclination, RAAN, argument of periapsis, true anomaly
     
     All values are taken to be in radians
     '''
    a,e,i,O,w,ta = oe
    return np.array([(a*(1-e**2))/(1+e*np.cos(ta)),
                     0,
                     0,
                     (mu/np.sqrt(mu*oe[0]*(1-oe[1]**2)))*oe[1]*np.sin(oe[5]),
                     (mu/np.sqrt(mu*oe[0]*(1-oe[1]**2)))*(1+oe[1]*np.cos(oe[5])),
                     0])

def rth_to_xyz(state,oe):
    ''' Converts a 6 element list or ndarray of the form: radial position, tangential position, 
    out-of-plane position, radial velocity, tangential velocity, out-of-plane velocity and a list
    or ndarray of the orbital elements into a cartesian state vector
     
     All values are taken to be in km or km/s
     '''
    raan, th, i = oe[3],oe[4]+oe[5],oe[2]
    cr = np.cos(raan)
    sr = np.sin(raan)
    ct = np.cos(th)
    st = np.sin(th)
    ci = np.cos(i)
    si = np.sin(i)
    mat = np.array([[cr*ct-sr*ci*st,    -cr*st-sr*ci*ct,    sr*si   ],
                    [sr*ct+cr*ci*st,    -sr*st+cr*ci*ct,    -cr*si  ],
                    [si*st,             si*ct,              ci      ]])
    if state.size == 6:
        mat = np.kron(np.eye(2),mat)
    return mat @ state

def EOM_2BP(x, t, mu):
    grav = -mu/LA.norm(x[0:3])**3
    x_dot = np.array([  x[3],   x[4],   x[5],   grav*x[0],  grav*x[1],   grav*x[2]])
    return x_dot

# ------------------------------------------------------------------------------------------------

URL = "http://127.0.0.1:5000/measurements/"

requests.delete(url=URL)

oe0 = [8000,0.001,0.001,0,0,0.25]
x0 = rth_to_xyz(oe_to_rth(oe0,wms.mu_Ea),oe0)
P0 = [1,1,1,1e-3,1e-3,1e-3]
pert0 = 1e-7*np.ones(6)

t0 = 58538
dt = 5

station1 = RRE.Station("Boulder", 40.0, -105.2, 5400)
station2 = RRE.Station("Melbourne",-37.8,144.95,100)
mission_id = 'The Coolest Mission in the World'

i = 0
x = x0
t = t0*86400
requests.put(url=URL,json={'stations':{station1.name:{'latitude':station1.latitude,
                              'longitude':station1.longitude,'height':station1.height},station2.name:{'latitude':station2.latitude,
                              'longitude':station2.longitude,'height':station2.height}},
                              'force_model':['point_mass', 'j2_accel'],'apriori_state':list(x0),
                              'apriori_cov':P0,'apriori_pert':list(pert0),'init_time':t})
while i < 120:
    k1 = EOM_2BP(x,t,wms.mu_Ea)
    k2 = dt * EOM_2BP(x+k1/2,t+dt/2,wms.mu_Ea)
    k3 = dt * EOM_2BP(x+k2/2,t+dt/2,wms.mu_Ea)
    k4 = dt * EOM_2BP(x+k3,t+dt,wms.mu_Ea)
    x = x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    x_s = station1.get_ECI(t/86400)
    t = t+dt
    range = np.sqrt((x[0]-x_s[0])**2+(x[1]-x_s[1])**2+(x[2]-x_s[2])**2)# + random.uniform(-0.01,0.01)
    params = {'type':'RangeMeas','value':range,'time':t,'mission_id':mission_id,'sigma':0.01,'station_id':station1.name}
    requests.post(url=URL,json=params)
    time.sleep(1)
    i += 1

