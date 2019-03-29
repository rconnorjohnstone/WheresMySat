#/usr/bin/env python
"""Database Module

Author: Hunter Mellema
Summary: Provides the ability to interface with and write kalman filter measurements to an 
    influx database.

"""
from influxdb import InfluxDBClient
import numpy as np
import os 

# Load Environment variables 
STATE_LIST = ["x_inertial", "y_inertial", "z_inertial", 
              "dx_inertial", "dy_inertial", "dz_inertial",
              "sigma_x", "sigma_y", "sigma_z", 
              "sigma_dx", "sigma_dy", "sigma_dz",
              "three_sig_x", "three_sig_y", "three_sig_z", 
              "three_sig_dx", "three_sig_dy", "three_sig_dz"]

# JSON Templates
STATE_TEMPLATE = {
    "measurement": "state_estimate",
    "time": None,
    "fields": {val:None for val in STATE_LIST}
}

# Database Client
class DataBaseIO(object):
    """Class to write KF estimates and covariance to a database

    Loads all environment variables from the container into a database 
    client object. This provides 

    All environment variables should be declared in the docker-compose.yml file
    at the root repository
    
    """
    HOST_IP = os.environ.get('INFLUX_HOST','filter-data'
    HOST_PORT = os.environ.get('INFLUX_PORT', 8086)
    USERNAME = os.environ.get('INFLUX_USER', 'worker-node')
    SECRET = os.environ.get('INFLUX_SECRET', 'foo')
    DATABASE = os.environ.get('INFLUX_DB', 'filter_data')
    
    def __init__(self):
        self.client = InfluxDBClient(host=self.HOST_IP,
                                     port=self.HOST_PORT,
                                     username=self.USERNAME,
                                     password=self.SECRET,
                                     database=self.DATABASE)

    def push(self, state_est, cov_est, time):
        """ Pushes kalman filter outputs to database

        Args:
            state_est (np.ndarray [1 x n]): state vector estimated by kalman filter
            cov_est (np.ndarray [n x n]): covariance matrix estimated by filter
            time (str): time at which measurement was taken

        """
        sigmas = parse_cov(cov_est, len(state_est))
        three_sigs = np.multiply(3, sigmas)

        # copies template for population
        params_json = STATE_TEMPLATE.copy()

        # populates template with data
        params_json['time'] = time
        for idx, val in enumerate(state_list):
            if idx in [0, 1, 2, 3, 4, 5]:
                params_json['fields'][val] = state_est[idx]
            if idx in [6, 7, 8]:
                params_json['fields'][val] = sigmas[idx-3]
            if idx in [9, 10, 11]:
                params_json['fields'][val] = three_sigs[idx-6]

        self.client.write_points([params_json])


def parse_cov(cov, len_state):
    """Finds the sigmas from the diagonals of the covariance matrix

    Args:
        cov (np.ndarray [n x n]): covariance matrix output from kalman filter
        len_state (int): length of the state vector

    """
    return  [np.sqrt(np.fabs(cov[i][i])) for i in range(len_state)]
