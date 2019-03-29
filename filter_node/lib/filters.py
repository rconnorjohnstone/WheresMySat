#!/usr/bin/env python
"""Filters Module

Author: Chelsea Thangavelu
Summary: Provides a framework for Kalman Filters for the WheresMySat project

Currently Implemented filters include:
- Classic Kalman Filter

"""
# standard library imports
from abc import ABCMeta, abstractmethod

# third party imports
from scipy.integrate import solve_ivp
import numpy as np
import ad
from ad import jacobian


class KF(object):
    """Parent class for Kalman Filters

    Args:
       state0_ref (np.ndarray [1 x n]): initial state vector estimate before measurment update
       P0 (np.ndarray [n x n]): Apriori covariance matrix
       dx_p (np.ndarray [1 x n]): initial perturbation vector
       t_prev (float): previous time
       force_model (WMS.ForceModel): force model to use for propagation

    """
    __metaclass__ = ABCMeta

    def __init__(self, state0_ref, P0, t_prev, force_model):
        self.state0_ref = state0_ref
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
        """computes Kalman Gain

        Args:
           P_m (np.ndarray [n x n]): post-propagtion, pre-measurement covariance
           R (float) = variance of measurement
           h_tilde (np.ndarray [p x n]): measurement sensitivity matrix

        Returns:
           (np.ndarray)

        """
        h_tilde = np.array(H_tilde)
        K = P_m @ h_tilde.T @ np.linalg.inv(h_tilde @ P_m @ h_tilde.T + R)

        return h_tilde, K

    def _compute_stm_and_state(self, state, time):
        """integrate EOM and A equation to solve for STM and state

        Args:
           state (np.ndarray [1 x n]): previous state estimate
           time (float): time to propagate stm and state to (usually measurement time)

        Returns:
           (np.ndarray [1 x n], np.ndarray [n x n])

        """
        phi = np.eye(self.len_state)
        state_w_phi = np.concatenate((state.reshape(self.len_state), phi.flatten()))
        state_w_phi = solve_ivp(self.state_w_phi_ode, [self.t_prev, time],
                                state_w_phi)
        state_w_phi = state_w_phi.y[:,-1]
        state = state_w_phi[:self.len_state]
        phi = state_w_phi[self.len_state:].reshape((self.len_state, self.len_state))

        return state, phi


    def _derivatives(self, state):
        """ Computes the jacobian and state derivatives

        Args:
            state (np.ndarray): state vector to find derivatives of

        Returns:
            (np.ndarray [1 x n], np.ndarray [n x n])

        """
        ad_state = ad.adnumber(state)
        state_deriv = self.force_model(0, ad_state)
        a_matrix = jacobian(state_deriv, ad_state)
        # remove ad number from all state values before returning
        state_deriv = [state.real for state in state_deriv]

        return state_deriv, a_matrix


    def state_w_phi_ode(self, t, state_w_phi):
        """ODE describing state dynamics and dphi/dt

        Args:
            t (float): time at which to evaluate ode (usually just a dummy)
            state_w_phi (np.ndarray [1 x (n^2 + n)]): flat array of state
                concatenated with the flattened phi matrix

        Returns:
            (np.ndarray [1 x (n^2 + n)])

        """
        #unpack phi and state
        phi = state_w_phi[self.len_state:].reshape((self.len_state,
                                                     self.len_state))
        state = state_w_phi[:self.len_state]
        #calculate derivative of phi and state
        state_deriv, a_matrix = self._derivatives(state)
        phid =  a_matrix @ phi
        phid_flat = phid.flatten()
        #concatenate state and phi derivatives
        state_w_phi_d = np.concatenate((state_deriv, phid_flat))

        return state_w_phi_d


class CKFilter(KF):
    """Classic Kalman Filter

    Basic kalman filter using a perturbation state and nonlinear state propagation.
    Inherits from KF (see for inputs)

    Args:
        kf_args: arguments for KF parent class
        dx_p (np.ndarray): initial guess of perturbation vector

    """
    def __init__(self, state0_ref, P0, dx_p, t_prev, force_model):
        super().__init__(state0_ref, P0, t_prev, force_model)
        self.dx_p = dx_p


    def run(self, msr):
        """Runs a single kalman filter update step given a measurement

        """
        state_ref, dx_m, P_m = self._timeupdate(self.state0_ref, msr.time, self.P0)
        resid, h_tilde = msr.residual(state_ref)
        h_tilde, K = self._kalman_gain(P_m, msr.sigma, h_tilde)
        dx_p, P_p = self._measupdate(dx_m, P_m, resid, h_tilde, K, msr.sigma)

        #store covariance and state corrections
        self.state0_ref = state_ref
        self.P0 = P_p
        self.dx_p = dx_p

        return state_ref + dx_p, P_p


    def _timeupdate(self, state, time, P_p):
        """Propagate state and covariance forward

        Args:
            state (np.ndarray [1 x n]): state vector
            time (float): time to propagate state and covariance to

        Returns:
            (np.ndarray [1 x n], np.ndarrray [1 x n], np.ndarray [n x n])

        """
        state_ref, phi = self._compute_stm_and_state(state, time)
        dx_m = phi @ self.dx_p
        P_m = phi @ P_p @ phi.T

        return state_ref, dx_m, P_m


    def _measupdate(self, dx_m, P_m, msr_resid, h_tilde, K, R):
        """Compute posteriori state and covariance

        Args:
           dx_m (np.ndarray [1 x n]): post propagation perturbation vector
           P_m (np.ndarray [n x n]): post propagation covariance matrix
           msr_resid (np.ndarray [q x 1]): measurement residuals
           h_tilde (np.ndarray [q x n]): measurement sensitivity matrix
           K (np.ndarray): Kalman gain
           R (float): measurement variance

        Returns:
           (np.ndarray [1 x n], np.ndarray [n x n])

        """
        dx_p = dx_m + K @ (msr_resid - h_tilde @ dx_m)
        P_p = (np.eye(self.len_state) - K @ h_tilde) @ P_m @ (np.eye(self.len_state)\
               - K @ h_tilde).T + R * K @ K.T

        return dx_p, P_p
