from influxdb import InfluxDBClient
import numpy as np
import time
import datetime

# Defaults
HOST_IP = '10.203.164.122'
HOST_PORT = 8086
USERNAME = 'grafana'
SECRET = 'secretpass'
DATABASE = 'influx'


state_list = ["x_inertial", "y_inertial", "z_inertial", "dx_inertial", "dy_inertial", "dz_inertial",
              "sigma_x", "sigma_y", "sigma_z", "sigma_dx", "sigma_dy", "sigma_dz",
              "three_sig_x", "three_sig_y", "three_sig_z", "three_sig_dx", "three_sig_dy", "three_sig_dz"]

# JSON Templates
STATE_TEMPLATE = {
    "measurement": "state_estimate",
    "time": None,
    "fields": {val:None for val in state_list}
}

# Database Client
class DataBaseIO(object):
    """Class to write KF estimates and covariance to a database

    Args:
        host (str): IP of database host
        port (int): port on which database is listening for HTTP
        username (str): username to
    """
    def __init__(self, host=HOST_IP, port=HOST_PORT, username=USERNAME,
                 password=SECRET, database=DATABASE):

        self.client = InfluxDBClient(host=HOST_IP,
                                     port=HOST_PORT,
                                     username=USERNAME,
                                     password=SECRET,
                                     database=DATABASE)

    def push(self, state_est, cov_est, time):
        """ Pushes kalman filter outputs to database
        """
        sigmas = parse_cov(cov_est, len(state_est))
        three_sigs = np.multiply(3, sigmas)

        # copies template for population
        params_json = STATE_TEMPLATE.copy()

        # populates template with data

        params_json['time'] = time
        for idx, val in enumerate(state_list):
            if idx in [0,1,2,3,4,5]:
                params_json['fields'][val] = state_est[idx]
            if idx in [6, 7, 8]:
                params_json['fields'][val] = sigmas[idx]
            if idx in [9, 10, 11]:
                params_json['fields'][val] = three_sigs[idx]

        self.client.write_points([params_json])


def parse_cov(cov, len_state):
    """Finds the sigmas from the diagonals of the covariance matrix

    Args:
        cov (np.ndarray [n x n]): covariance matrix output from kalman filter
        len_state (int): length of the state vector


    """
    return  [np.sqrt(np.fabs(cov[i][i])) for i in range(len_state)]
