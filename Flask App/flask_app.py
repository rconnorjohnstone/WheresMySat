#!/usr/bin/env python
''' The dreaded flask app......

Author:Connor Johnstone
'''

from flask import Flask
from flask import request
import csv

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'



@app.route('/hello',methods=['GET','POST'])
def hello_world():
    return request.method

@app.route('/measurements/',methods=['GET','POST','PUT'])
def grab_measurements():
    if request.method == 'PUT':
        
        kf = CKFilter()

    elif request.method == 'POST':
        try: 
            kf.mease
        except: 
            raise ValueError("You suck")
            
            
        with open('measurements.txt', mode='a') as csv_file:
            meas_writer = csv.writer(csv_file)
            meas_writer.writerow([request.args.to_dict(flat=False)])
        return 'Thanks!' 
    else:
        return "Send Me Your Measurements!!!"
