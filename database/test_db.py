from influxdb import InfluxDBClient
import numpy as np
import time
import datetime

# set time


# HTTP connection
client = InfluxDBClient(host='10.203.164.122', port=8086, username='grafana',
                        password='secretpass', database='influx')

## template to use for estimated recording
json_template = {
    "measurement": "state_estimate",
    "tags": {
        "filter": None
    },
    "time": None,
    "fields": {
        "x_inertial": None,
        "y_inertial": None,
        "z_inertial": None
    }
}

### generate points
times = np.arange(0,100, 10)
q = np.random.rand(len(times),3)

### generate data
json_list = []
for idx, val in enumerate(q):
    time.sleep(1)
    json_val = json_template.copy()
    now = datetime.datetime.now()
    json_val['time'] = now.strftime(
        '%Y-%m-%dT%H:%M:%SZ'
    )
    print(json_val['time'])
    json_val['fields']["x_inertial"] = val[0]
    json_val['fields']["y_inertial"] = val[1]
    json_val['fields']["z_inertial"] = val[2]
    json_list.append(json_val)


### Push to database
client.write_points(json_list)
