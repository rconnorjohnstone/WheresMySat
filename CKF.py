import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from propagator import *
import ad
from ad import jacobian
from FakeMeasurements import genMSM

Re = 6378.1363 #km
mu = 3.986004415e5 #km^3/s^2
J2 = 0.0010826269 #unitless
J3 = 0   #we are not modeling J3

class KF(object):
    """Parent Filter Class :)"""
    __metaclass__ = ABCMeta

    def __init__(self, state0_ref, P0, dx_p, t_prev, force_model):
        self.state0_ref = state0_ref
        self.states_ref = [state0_ref]
        self.P0 = P0
        self.t_prev = t_prev
        self.force_model = force_model
        self.len_state = len(state0_ref)
        
    @abstractmethod
    def run(self):
        """conglomerate of calls to functions to follow (e.g. _timeupdate)"""
        pass
    
    @abstractmethod
    def _timeupdate(self):
        """propagate state and covariance forward"""
        pass
        
    @abstractmethod
    def _measupdate(self):
        """compute posteriori covariance and state correction"""
        pass
    
    def _kalman_gain(self, P_m, R, H_tilde):
        """compute Kalman Gain"""
        K = P_m @ H_tilde.T @ np.linalg.inv(H_tilde @ P_m @ H_tilde.T + R)
        
        return H_tilde, K

    def _compute_stm_and_state(self, state, time):
        """integrate EOM and A equation to solve for STM and state"""
        phi = np.eye(self.len_state)
        state_w_phi = np.concatenate((state.reshape(self.len_state), phi.flatten()))
        state_w_phi = solve_ivp(self.state_w_phi_ode, [self.t_prev, time], \
                                state_w_phi)
        state_w_phi = state_w_phi.y[:,-1]
        state = state_w_phi[:self.len_state]
        phi = state_w_phi[self.len_state:].reshape((self.len_state, self.len_state))
        
        return state, phi   
    
    def _derivatives(self, state):
        """ Computes the jacobian and state derivatives

        Args:
            state (np.ndarray): state vector to find derivatives of

        """
        ad_state = ad.adnumber(state)
        state_deriv = self.force_model(0, ad_state)
        a_matrix = jacobian(state_deriv, ad_state)

        return state_deriv, a_matrix
    
    def state_w_phi_ode(self, t, state_w_phi):
        """ODE describing state dynamics and dphi/dt"""
        n = self.len_state
        #unpack phi and state
        phi = state_w_phi[n:].reshape((n,n))
#        phi = np.eye(self.len_state)
        state = state_w_phi[:n]
        #calculate derivative of phi and state
        state_deriv, a_matrix = self._derivatives(state)
        phid =  a_matrix @ phi
        phid_flat = phid.flatten()
        #concatenate state and phi derivatives
        state_w_phi_d = np.concatenate((state_deriv, phid_flat))
        
        return state_w_phi_d
        
class CKFilter(KF):
    """CKF :)
    
    """
    def __init__(self, state0_ref, P0, dx_p, t_prev, force_model, msr):
        super().__init__(state0_ref, P0, dx_p, t_prev, force_model)
        self.dx_p = dx_p 
        
    def run(self, msr):
#        for msr in self.msrs:
        #time update
        state_ref, dx_m, P_m = self._timeupdate(self.state0_ref, msr.time, self.P0)
        #observation
        H_tilde, K = self._kalman_gain(P_m, msr.sigma, msr.H_tilde)
        #measurment update
        dx_p, P_p = self._measupdate(dx_m, P_m, msr.r, msr.H_tilde, K, msr.sigma)
        #store covariance and state corrections
        self.state0_ref = state_ref
        self.P0 = P_p
        self.dx_p = dx_p
        
    #propagate state and covariance forward            
    def _timeupdate(self, state, time, P_p):
        state_ref, phi = self._compute_stm_and_state(state, time)
        dx_m = phi @ self.dx_p
        P_m = phi @ P_p.T @ phi.T
        return state_ref, dx_m, P_m
    
    
    #compute posteriori state and covariance
    def _measupdate(self, dx_m, P_m, r, H_tilde, K, R):
        dx_p = dx_m + K @ (r - H_tilde @ dx_m)
        P_p = (np.eye(self.len_state) - K @ H_tilde) @ P_m @ (np.eye(self.len_state)\
               - K @ H_tilde).T + R * K @ K.T
        return dx_p, P_p
    

state0_ref = np.array([-3515.49032703351, 8390.716310243391, 4127.627352553682, \
              -4.357676323818018, -3.356579140027686, 3.1118929290409585]).reshape((6,1))
msrs = genMSM()
P0 = np.diag(np.full(6,1e2))
dx_p = np.zeros(6)
R = 1e-6#np.diag([1e-6,1e-9])
len_state = 6
xs = []
ckfest = [dx_p+state0_ref]
dx_ps = [dx_p]
P_s = [P0]
t_prev = 0
ts = [t_prev]
for index in range(len(msrs)):
    if index==0:
        t_prev = 0
    else:
        t_prev = msrs[index].time
        state0_ref = x.state0_ref
        P0 = x.P0
        t_prev = msrs[index - 1].time
    ts.append(msrs[index].time)
    x = CKFilter(state0_ref, P0, dx_p, t_prev, \
                       ForceModel([point_mass,j2_accel]), msrs[index])
    x.run(msr)
    ckfest.append(x.state0_ref+x.dx_p.reshape((6,1)))
    P_s.append(x.P0)
    dx_ps.append(x.dx_p)
    xs.append(x)
    print(x.dx_p)

#plot for fun
plotselect = [1]
#ts = np.linspace(0,69,70)
if 1 in plotselect:
    P_ps = np.array(P_s)
    dx_p = np.array(dx_ps)
    fig, ax = plt.subplots(3,2)
    for j in range(2):
        for i in range(3):
            ax[i,j].scatter(ts[:], dx_p[:,i+j])
            ax[i,j].scatter(ts[:], P_ps[:,i+j,i+j], s=2)