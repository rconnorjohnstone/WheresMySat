#!/usr/bin/env python
''' Definition of the measurement class for the CU Hackathon Project: WheresMySat

Parent class meas records a measurement to be fed into the Kalman Filter

All children classes of meas contain any extraneous data and the system dynamics

Author:Connor Johnstone
'''

#Import Libraries
from abc import ABC, abstractmethod
import ad

class Meas(ABC):
    ''' Measurement Class
    
    Records all data common to all measurements our Kalman Filter should expect to see
    
    Arguments:
         - value: (float64) Measurement value
         - sigma: (float64) Measurement variance
         - time: (float64) timestamp of measurements in seconds from reference epoch
         - mission_id: (string) name of mission to match records to
         
    Measurement class is generally considered incomplete without an additional child class for
    the measurement type
    '''
    def __init__(self,value,time,mission_id,sigma):
        self.value = value
        self.sigma = sigma
        self.t = time
        self.mis_id = mission_id
        
    @abstractmethod
    def gen_meas(self):
        ''' The method used to generate a measurement from expected state data
        
        This method must be overwritten by the child class
        '''
        pass
    
    def residual(self,state):
        ad_state = ad.adnumber(state)
        est_msr = self.gen_meas(ad_state)
        resid = self.value - est_msr.real
        H_tilde = ad.jacobian(est_msr, ad_state)
        return resid,H_tilde
    
        

        