# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 22:11:40 2019

@author: chels
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 17:21:20 2019

@author: chels
"""

import numpy as np
from coe2rv import coe2rvfunc, rad2deg, deg2rad
from math import sqrt
from scipy.integrate import solve_ivp


def constants():
    Re = 6378.0 #km
    mu = 3.986004415e5 #km^3/s^2
    J2 = 0.0010826269
    J3 = -2.53241051856772e-6
    return Re, mu, J2

def genwe():
    we = 7.29211585532-5
    return we

def eom(t,rv):
    Re, mu, J2 = constants()
    x,y,z,u,v,w = rv
    r = np.sqrt(x**2+y**2+z**2)
    xdot = u
    vxdot = -mu*x/(r**3)-3.*J2*mu*Re**2*x*(1.-5.*z**2/(r**2))/(2*r**5)
    ydot = v
    vydot = -mu*y/(r**3)-3.*J2*mu*Re**2*y*(1.-5.*z**2/(r**2))/(2*r**5)
    zdot = w
    vzdot = -mu*z/(r**3)-3.*J2*mu*Re**2*z*(3.-5.*z**2/(r**2))/(2.*r**5)
    
    drvdt=[xdot,ydot,zdot,vxdot,vydot,vzdot]
    return drvdt


def dfdx_wJ2(w):
    Re, mu, J2 = constants()
    x = w[0]
    y = w[1]
    z = w[2]
    J3 = 0.
    #f1 partials
    df1dx = (3*mu*x**2)/(x**2 + y**2 + z**2)**(5/2) - mu/(x**2 + y**2 + z**2)**(3/2) \
        - (5*J3*Re**3*mu*(3*z - (7*z**3)/(x**2 + y**2 + z**2)))/(2*(x**2 + y**2 + z**2)**(7/2)) \
        + (3*J2*Re**2*mu*((5*z**2)/(x**2 + y**2 + z**2) - 1))/(2*(x**2 + y**2 + z**2)**(5/2)) \
        + (35*J3*Re**3*mu*x**2*(3*z - (7*z**3)/(x**2 + y**2 + z**2)))/(2*(x**2 + y**2 + z**2)**(9/2)) \
        - (15*J2*Re**2*mu*x**2*z**2)/(x**2 + y**2 + z**2)**(9/2) \
        - (35*J3*Re**3*mu*x**2*z**3)/(x**2 + y**2 + z**2)**(11/2) \
        - (15*J2*Re**2*mu*x**2*((5*z**2)/(x**2 + y**2 + z**2) - 1))/(2*(x**2 + y**2 + z**2)**(7/2))
    df1dy = (3.*mu*x*y)/(x**2 + y**2 + z**2)**(5./2.) + \
        (35.*J3*Re**3.*mu*x*y*(3.*z - (7.*z**3)/(x**2 + y**2 + z**2)))/(2.*(x**2 + y**2 + z**2)**(9/2)) \
        - (15.*J2*Re**2*mu*x*y*z**2)/(x**2 + y**2 + z**2)**(9/2) \
        - (35.*J3*Re**3.*mu*x*y*z**3)/(x**2 + y**2 + z**2)**(11/2) \
        - (15.*J2*Re**2.*mu*x*y*((5.*z**2)/(x**2 + y**2 + z**2) - 1.))/(2.*(x**2 + y**2 + z**2)**(7/2))
    df1dz = (3*mu*x*z)/(x**2 + y**2 + z**2)**(5/2) + (3*J2*Re**2*mu*x*((10*z)/(x**2 + y**2 + z**2) \
        - (10*z**3)/(x**2 + y**2 + z**2)**2))/(2*(x**2 + y**2 + z**2)**(5/2)) \
        - (5*J3*Re**3*mu*x*((14*z**4)/(x**2 + y**2 + z**2)**2  \
        - (21*z**2)/(x**2 + y**2 + z**2) + 3))/(2*(x**2 + y**2 + z**2)**(7/2)) \
        + (35*J3*Re**3*mu*x*z*(3*z - (7*z**3)/(x**2 + y**2 + z**2)))/(2*(x**2 + y**2 + z**2)**(9/2)) \
        - (15*J2*Re**2*mu*x*z*((5*z**2)/(x**2 + y**2 + z**2) - 1))/(2*(x**2 + y**2 + z**2)**(7/2))
    df1dvx = 0
    df1dvy = 0
    df1dvz = 0
    #f2 partials
    df2dx = (1/(2 * (x**2 + y**2 + z**2)**( \
        11/2))) * 3 * mu * x * y * (2 * x**6 + 2 * y**6 + 35 * J3 * Re**3 * y**2 * z + 6 * y**4 * z**2 -  \
        70 * J3 * Re**3 * z**3 + 6 * y**2 * z**4 + 2 * z**6 + 6 * x**4 * (y**2 + z**2) +  \
        x**2 * (6 * y**4 + 35 * J3 * Re**3 * z + 12 * y**2 * z**2 + 6 * z**4) +  \
        5 * J2 * Re**2 * (x**4 + y**4 - 5 * y**2 * z**2 - 6 * z**4 + x**2 * (2 * y**2 - 5 * z**2)))
    df2dy = (3*mu*y**2)/(x**2 + y**2 + z**2)**(5/2) - mu/(x**2 + y**2 + z**2)**(3/2) -  \
        (5*J3*Re**3*mu*(3*z - (7*z**3)/(x**2 + y**2 + z**2)))/(2*(x**2 + y**2 + z**2)**(7/2)) +  \
        (3*J2*Re**2*mu*((5*z**2)/(x**2 + y**2 + z**2) - 1))/(2*(x**2 + y**2 + z**2)**(5/2)) +  \
        (35*J3*Re**3*mu*y**2*(3*z - (7*z**3)/(x**2 + y**2 + z**2)))/(2*(x**2 + y**2 + z**2)**(9/2)) -  \
        (15*J2*Re**2*mu*y**2*z**2)/(x**2 + y**2 + z**2)**(9/2) -  \
        (35*J3*Re**3*mu*y**2*z**3)/(x**2 + y**2 + z**2)**(11/2) -  \
        (15*J2*Re**2*mu*y**2*((5*z**2)/(x**2 + y**2 + z**2) - 1))/(2*(x**2 + y**2 + z**2)**(7/2))
    df2dz = (3*mu*y*z)/(x**2 + y**2 + z**2)**(5/2) + (3*J2*Re**2*mu*y*((10*z)/(x**2 + y**2 + z**2) \
        - (10*z**3)/(x**2 + y**2 + z**2)**2))/(2*(x**2 + y**2 + z**2)**(5/2)) \
        - (5*J3*Re**3*mu*y*((14*z**4)/(x**2 + y**2 + z**2)**2 -  \
        (21*z**2)/(x**2 + y**2 + z**2) + 3))/(2*(x**2 + y**2 + z**2)**(7/2)) +  \
        (35*J3*Re**3*mu*y*z*(3*z - (7*z**3)/(x**2 + y**2 + z**2)))/(2*(x**2 + y**2 + z**2)**(9/2)) -  \
        (15*J2*Re**2*mu*y*z*((5*z**2)/(x**2 + y**2 + z**2) - 1))/(2*(x**2 + y**2 + z**2)**(7/2))
    df2dvx = 0.
    df2dvy = 0.
    df2dvz = 0.
    #f3 partials
    df3dx = (3*mu*x*z)/(x**2 + y**2 + z**2)**(5/2) + (5*J3*Re**3*mu*((6*x)/5 - (14*x*z**4)/ \
        (x**2 + y**2 + z**2)**2))/(2*(x**2 + y**2 + z**2)**(7/2)) - (35*J3*Re**3*mu*x*((7*z**4)/ \
        (x**2 + y**2 + z**2) + (3*x**2)/5 + (3*y**2)/5 - (27*z**2)/5))/(2*(x**2 + y**2 + z**2)**(9/2)) \
        - (15*J2*Re**2*mu*x*z**3)/(x**2 + y**2 + z**2)**(9/2) -  \
        (15*J2*Re**2*mu*x*z*((5*z**2)/(x**2 + y**2 + z**2) - 3))/(2*(x**2 + y**2 + z**2)**(7/2))
    df3dy = (3*mu*y*z)/(x**2 + y**2 + z**2)**(5/2) +  \
        (5*J3*Re**3*mu*((6*y)/5 - (14*y*z**4)/(x**2 + y**2 + z**2)**2))/(2*(x**2 + y**2 + z**2)**(7/2)) \
        - (35*J3*Re**3*mu*y*((7*z**4)/(x**2 + y**2 + z**2) + (3*x**2)/5 + (3*y**2)/5  \
    - (27*z**2)/5))/(2*(x**2 + y**2 + z**2)**(9/2)) - (15*J2*Re**2*mu*y*z**3)/(x**2 + y**2 + z**2)**(9/2)  \
        - (15*J2*Re**2*mu*y*z*((5*z**2)/(x**2 + y**2 + z**2) - 3))/(2*(x**2 + y**2 + z**2)**(7/2))
    df3dz = (3*mu*z**2)/(x**2 + y**2 + z**2)**(5/2) - mu/(x**2 + y**2 + z**2)**(3/2) -  \
        (5*J3*Re**3*mu*((54*z)/5 - (28*z**3)/(x**2 + y**2 + z**2) +  \
        (14*z**5)/(x**2 + y**2 + z**2)**2))/(2*(x**2 + y**2 + z**2)**(7/2)) +  \
        (3*J2*Re**2*mu*((5*z**2)/(x**2 + y**2 + z**2) - 3))/(2*(x**2 + y**2 + z**2)**(5/2)) -  \
        (35*J3*Re**3*mu*z*((7*z**4)/(x**2 + y**2 + z**2) + (3*x**2)/5 + (3*y**2)/5 -  \
        (27*z**2)/5))/(2*(x**2 + y**2 + z**2)**(9/2)) + (3*J2*Re**2*mu*z*((10*z)/(x**2 + y**2 + z**2) -  \
        (10*z**3)/(x**2 + y**2 + z**2)**2))/(2*(x**2 + y**2 + z**2)**(5/2)) -  \
        (15*J2*Re**2*mu*z**2*((5*z**2)/(x**2 + y**2 + z**2) - 3))/(2*(x**2 + y**2 + z**2)**(7/2))
    df3dvx = 0
    df3dvy = 0
    df3dvz = 0
#    R = Re
#    df1dx = -(2*mu*(x**2 + y**2 + z**2)**4 - 6*mu*x**2*(x**2 + y**2 + z**2)**3 + 3*J2*R**2*mu*(x**2 + y**2 + z**2)**3 + 315*J3*R**3*mu*x**2*z**3 + 15*J3*R**3*mu*z*(x**2 + y**2 + z**2)**2 - 35*J3*R**3*mu*z**3*(x**2 + y**2 + z**2) - 15*J2*R**2*mu*x**2*(x**2 + y**2 + z**2)**2 - 15*J2*R**2*mu*z**2*(x**2 + y**2 + z**2)**2 - 105*J3*R**3*mu*x**2*z*(x**2 + y**2 + z**2) + 105*J2*R**2*mu*x**2*z**2*(x**2 + y**2 + z**2))/(2*(x**2 + y**2 + z**2)**(11/2))
#    df1dy = (6*mu*x*y*(x**2 + y**2 + z**2)**3 + 15*J2*R**2*mu*x*y*(x**2 + y**2 + z**2)**2 - 315*J3*R**3*mu*x*y*z**3 + 105*J3*R**3*mu*x*y*z*(x**2 + y**2 + z**2) - 105*J2*R**2*mu*x*y*z**2*(x**2 + y**2 + z**2))/(2*(x**2 + y**2 + z**2)**(11/2))
#    df1dy = (6*mu*x*y*(x**2 + y**2 + z**2)**3 + 15*J2*R**2*mu*x*y*(x**2 + y**2 + z**2)**2 - 315*J3*R**3*mu*x*y*z**3 + 105*J3*R**3*mu*x*y*z*(x**2 + y**2 + z**2) - 105*J2*R**2*mu*x*y*z**2*(x**2 + y**2 + z**2))/(2*(x**2 + y**2 + z**2)**(11/2))
#    df2dx = (6*mu*x*y*(x**2 + y**2 + z**2)**3 + 15*J2*R**2*mu*x*y*(x**2 + y**2 + z**2)**2 - 315*J3*R**3*mu*x*y*z**3 + 105*J3*R**3*mu*x*y*z*(x**2 + y**2 + z**2) - 105*J2*R**2*mu*x*y*z**2*(x**2 + y**2 + z**2))/(2*(x**2 + y**2 + z**2)**(11/2))
#    df2dy = -(2*mu*(x**2 + y**2 + z**2)**4 - 6*mu*y**2*(x**2 + y**2 + z**2)**3 + 3*J2*R**2*mu*(x**2 + y**2 + z**2)**3 + 315*J3*R**3*mu*y**2*z**3 + 15*J3*R**3*mu*z*(x**2 + y**2 + z**2)**2 - 35*J3*R**3*mu*z**3*(x**2 + y**2 + z**2) - 15*J2*R**2*mu*y**2*(x**2 + y**2 + z**2)**2 - 15*J2*R**2*mu*z**2*(x**2 + y**2 + z**2)**2 - 105*J3*R**3*mu*y**2*z*(x**2 + y**2 + z**2) + 105*J2*R**2*mu*y**2*z**2*(x**2 + y**2 + z**2))/(2*(x**2 + y**2 + z**2)**(11/2))
#    df2dz = (6*mu*y*z*(x**2 + y**2 + z**2)**3 - 315*J3*R**3*mu*y*z**4 - 15*J3*R**3*mu*y*(x**2 + y**2 + z**2)**2 + 45*J2*R**2*mu*y*z*(x**2 + y**2 + z**2)**2 - 105*J2*R**2*mu*y*z**3*(x**2 + y**2 + z**2) + 210*J3*R**3*mu*y*z**2*(x**2 + y**2 + z**2))/(2*(x**2 + y**2 + z**2)**(11/2))
#    df3dx = (6*mu*x*z*(x**2 + y**2 + z**2)**3 - 315*J3*R**3*mu*x*z**4 - 15*J3*R**3*mu*x*(x**2 + y**2 + z**2)**2 + 45*J2*R**2*mu*x*z*(x**2 + y**2 + z**2)**2 - 105*J2*R**2*mu*x*z**3*(x**2 + y**2 + z**2) + 210*J3*R**3*mu*x*z**2*(x**2 + y**2 + z**2))/(2*(x**2 + y**2 + z**2)**(11/2))
#    df3dy = (6*mu*y*z*(x**2 + y**2 + z**2)**3 - 315*J3*R**3*mu*y*z**4 - 15*J3*R**3*mu*y*(x**2 + y**2 + z**2)**2 + 45*J2*R**2*mu*y*z*(x**2 + y**2 + z**2)**2 - 105*J2*R**2*mu*y*z**3*(x**2 + y**2 + z**2) + 210*J3*R**3*mu*y*z**2*(x**2 + y**2 + z**2))/(2*(x**2 + y**2 + z**2)**(11/2))
#    df3dz = -(2*mu*(x**2 + y**2 + z**2)**4 - 6*mu*z**2*(x**2 + y**2 + z**2)**3 + 9*J2*R**2*mu*(x**2 + y**2 + z**2)**3 + 315*J3*R**3*mu*z**5 + 105*J2*R**2*mu*z**4*(x**2 + y**2 + z**2) + 75*J3*R**3*mu*z*(x**2 + y**2 + z**2)**2 - 350*J3*R**3*mu*z**3*(x**2 + y**2 + z**2) - 90*J2*R**2*mu*z**2*(x**2 + y**2 + z**2)**2)/(2*(x**2 + y**2 + z**2)**(11/2))
    dfdx = np.array([[0.,0., 0., 1., 0., 0.],\
        [0., 0., 0., 0., 1., 0.],\
        [0., 0., 0., 0., 0., 1.],\
        [df1dx, df1dy, df1dz, df1dvx, df1dvy, df1dvz],\
        [df2dx, df2dy, df2dz, df2dvx, df2dvy, df2dvz],\
        [df3dx, df3dy, df3dz, df3dvx, df3dvy, df3dvz]])
    return dfdx
    

def keplerJ2_wPhi_ODE_thangavelu(t,z): #[time,state]
    Re,mu,J2 = constants()
    n = 6
    xs = z[0:n] #state
    phi = z[n:n**2+n].reshape((n,n)).T
    #use EOM to calculate xd
    xd = eom(t,xs[0:6])
    #calculate dfdz = A
    dfdx = dfdx_wJ2(xs)
    #calculate phid
    phid = dfdx @ phi
    phid_flat = phid.T.flatten()
    zd = np.concatenate((np.array(xd).reshape(len(xd)),phid_flat))
    return zd

    

# testing
#y0 = list(coe2rvfunc(10000., .001, deg2rad(40.), deg2rad(80.), deg2rad(40.), \
#                     0.).reshape(6))
#z0 = np.concatenate((y0, np.eye(6).flatten()))
#tf = 10000
#ts = np.arange(0, tf, 10)
#sol = solve_ivp(keplerJ2_wPhi_ODE_thangavelu, [0., tf], z0, t_eval = ts, \
#                max_step = 5)
##compare to hwtraj1.txt at 300 seconds
#phis = sol.y[6:,:]
#phitest = phis[:,30].reshape((6,6))