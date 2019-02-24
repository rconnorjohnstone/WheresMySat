import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Re = 6378.1363 #km
mu = 3.986004415e5 #km^3/s^2
J2 = 0.0010826269 #unitless
J3 = 0   #we are not modeling J3

    
class CKFilter:
    
    def __init__(self, state0_ref, msrs, P0, dx_p, R, len_state):
        self.state0_ref = state0_ref
        self.states_ref = [state0_ref]
        self.msrs = msrs
        self.P0 = P0
        self.P_ps = [P0]
        self.dx_p = [dx_p]
        self.len_state = len_state
        self.R = R
        self.times = [0]
        
    def run(self):
        for msr in self.msrs:
            #time update
            state_ref, dx_m, P_m = self._timeupdate(msr.time)
            #observation
            H_tilde, K = self._obs(state_ref, P_m, self.R, msr.H_tilde)
            #measurment update
            dx_p, P_p = self._measupdate(dx_m, P_m, msr.r, msr.H_tilde, K)
            #store covariance and state corrections
            self.times.append(msr.time)
            self.states_ref.append(state_ref)
            self.dx_p.append(dx_p)
            self.P_ps.append(P_p)
                
    def _timeupdate(self, time):
        state_ref, phi = self._compute_stm_and_state(time)
        dx_m = phi @ self.dx_p[-1]
        P_m = phi @ self.P_ps[-1].T @ phi.T
        return state_ref, dx_m, P_m
    
    def _obs(self, state_prop, P_m, R, H_tilde):
        K = P_m @ H_tilde.T @ np.linalg.inv(H_tilde @ P_m @ H_tilde.T + R)
#        K = P_m @ H_tilde.T * (H_tilde @ P_m @ H_tilde.T + R)**(-1)
        return H_tilde, K
    
    def _measupdate(self, dx_m, P_m, r, H_tilde, K):
        dx_p = dx_m + K @ (r - H_tilde @ dx_m)
        P_p = (np.eye(self.len_state) - K @ H_tilde) @ P_m @ (np.eye(self.len_state)\
               - K @ H_tilde).T + self.R * K @ K.T
        return dx_p, P_p
        
    
    def _compute_stm_and_state(self, time):#, phi = np.array([])):
#        if not phi.any():
        phi = np.eye(self.len_state)
        state_w_phi = np.concatenate((self.states_ref[-1][:,0], phi.flatten()))
        state_w_phi = solve_ivp(self._phi_ode, [self.times[-1], time], \
                                state_w_phi, t_eval = [time], max_step = 5).y
        state = state_w_phi[:self.len_state]
        phi = state_w_phi[self.len_state:].reshape((self.len_state, self.len_state))
        return state, phi
        
    def _phi_ode(self, t, state_w_phi):
        n = self.len_state
        #unpack phi and state
        phi = state_w_phi[n:].reshape((n,n)).T
        #calculate derivative of phi
        phid = phi @ self.dfdx_wJ2()
        phid_flat = phid.T.flatten()
        #calculate derivative of state
        stated = self.eom(t)
        #concatenate state and phi derivatives
        state_w_phi_d = np.concatenate((stated, phid_flat))
        return state_w_phi_d
        
    def eom(self, t):
        x,y,z,u,v,w = self.states_ref[-1][:,0]
        r = np.sqrt(x**2+y**2+z**2)
        xdot = u
        vxdot = -mu*x/(r**3)-3.*J2*mu*Re**2*x*(1.-5.*z**2/(r**2))/(2*r**5)
        ydot = v
        vydot = -mu*y/(r**3)-3.*J2*mu*Re**2*y*(1.-5.*z**2/(r**2))/(2*r**5)
        zdot = w
        vzdot = -mu*z/(r**3)-3.*J2*mu*Re**2*z*(3.-5.*z**2/(r**2))/(2.*r**5)
        drvdt=[xdot,ydot,zdot,vxdot,vydot,vzdot]
        return drvdt
        
    def dfdx_wJ2(self): #partials (A matrix)
        x, y, z, vx, vy, vz = self.states_ref[-1]
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
        dfdx = np.array([[0.,0., 0., 1., 0., 0.],\
            [0., 0., 0., 0., 1., 0.],\
            [0., 0., 0., 0., 0., 1.],\
            [df1dx, df1dy, df1dz, df1dvx, df1dvy, df1dvz],\
            [df2dx, df2dy, df2dz, df2dvx, df2dvy, df2dvz],\
            [df3dx, df3dy, df3dz, df3dvx, df3dvy, df3dvz]])
        return dfdx

state0_ref = np.array([-3515.49032703351, 8390.716310243391, 4127.627352553682, \
              -4.357676323818018, -3.356579140027686, 3.1118929290409585]).reshape((6,1))
msrs = msm
P0 = np.diag(np.full(6,1e-3))
dx_p = np.zeros(6)
R = 1e-6#np.diag([1e-6,1e-9])
len_state = 6
#CKFilter(state0_ref, msrs, P0, dx_p, R, len_state)
x = CKFilter(state0_ref, msrs, P0, dx_p, R, len_state)
x.run()

plotselect = [1]

if 1 in plotselect:
    P_ps = np.array(x.P_ps).shape
    dx_p = np.array(x.dx_p)
    fig, ax = plt.subplots(3,2)
    for j in range(2):
        for i in range(3):
            ax[i,j].scatter(x.times, dx_p[:,i+j])
            ax[i,j].scatter(x.times, P_ps[:,i+j,i+j])
    
