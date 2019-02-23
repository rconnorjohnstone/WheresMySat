''' Definition of the measurement class for the CU Hackathon Project: WheresMySat

Parent class meas records a measurement to be fed into the Kalman Filter

All children classes of meas contain any extraneous data and the system dynamics

Author:Connor Johnstone
'''

#Import Libraries
import numpy as np
from numpy import linalg as LA

class Meas():
    ''' Measurement Class
    
    Records all data common to all measurements our Kalman Filter should expect to see
    
    Arguments:
         - value: (list of np.ndarray) of measurement values
         - time: (float64) timestamp of measurements in seconds from reference epoch
         - mission_id: (string) name of mission to match records to
         
    Measurement class is generally considered incomplete without an additional child class for
    the measurement type
    '''
    def __init__(self,value,time,mission_id):
        self.val = value
        self.t = time
        self.mis_id = mission_id
    
        

        