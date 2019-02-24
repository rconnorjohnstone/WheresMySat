#!/usr/bin/env python
''' The dreaded flask app......

Author:Connor Johnstone
'''

from flask import Flask
from flask import request
import csv
import numpy as np
import WheresMySat as wms
from WheresMySat.propagator import *
from WheresMySat.filters import *
from WheresMySat.RRE_measurements import *
from WheresMySat.interface import *
from datetime import datetime
import julian

#from WheresMySat.filters import CKFilter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

@app.route('/hello',methods=['GET','POST'])
def hello_world():
    return request.method

@app.route('/measurements/',methods=['GET','POST','PUT','DELETE'])
def grab_measurements():
    if request.method == 'PUT':
        inputs = request.json
        forceModel = ForceModel([globals()[force] for force in inputs['force_model']])
        globals()['STATIONS'] = {station: Station(station, 
                                    inputs['stations'][station]['latitude'],
                                    inputs['stations'][station]['longitude'],
                                    inputs['stations'][station]['height'])  
                                    for station in inputs['stations']}
        apriori_state = np.array(inputs['apriori_state'])
        apriori_cov = np.diag(inputs['apriori_cov'])
        apriori_pert = np.array(inputs['apriori_pert'])
        t0 = inputs['init_time']
        globals()['KALMANFILTER'] = CKFilter(apriori_state+apriori_pert,apriori_cov,apriori_pert,t0,forceModel)
        globals()['DBIO'] = DataBaseIO()
        return 'ridiculous'
    elif request.method == 'POST':
        inputs = request.json
        value = inputs['value']
        time = inputs['time']
        mission_id = inputs['mission_id']
        sigma = inputs['sigma']
        station = globals()['STATIONS'][inputs['station_id']]
        measure = globals()[inputs['type']](value,time,mission_id,station,sigma)
        state_est, cov_est = globals()['KALMANFILTER'].run(measure)
        
        time = time/86400
        dt = julian.from_jd(time,fmt='mjd')
        time = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        globals()['DBIO'].push(state_est, cov_est, time)
        
        return 'something stupid'
    elif request.method == 'DELETE':
        print('Delete is running')
        globals()['DBIO'].client.drop_database('influx')
        globals()['DBIO'].client.create_database('influx')
    else:
        return "Send Me Your Measurements!!!"
