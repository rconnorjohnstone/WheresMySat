#!/usr/bin/env python
''' Definition of the measurement class specific to range, range rate, and elevation measurements

Parent class meas records a measurement to be fed into the Kalman Filter

All children classes of meas contain any extraneous data and the system dynamics

station class is used for range, rangerate measurements taken from AWS groundstation, defines
a station (latitude and longitude)

Author:Connor Johnstone
'''

#Import Libraries
import numpy as np
import WheresMySat as WMS
from WheresMySat.measurements import Meas
import astropy.coordinates as astro
from astropy import units as u
from astropy import time
import ad
from ad.admath import atan


class RangeMeas(Meas):
    ''' Range Measurement
    
    Includes information common to all measurements plus addition info (station_obj) specific to
    Range measurements
    
    Arguments:
        - station_obj: (station) Station object from Amazon groundstation (may be expanded in the
        future) which includes latitude, longitude of station
    '''
    def __init__(self,value,time,mission_id,station_obj,sigma=1):
        super(RangeMeas,self).__init__(value,time,mission_id,sigma)
        self.station_obj = station_obj
        
    def __repr__(self):
        return "Range is {} km from {} Station".format(self.value, self.station_obj.name)
        
    def gen_meas(self, state):
        x,y,z = state[0:3]
        x_s,y_s,z_s = self.station_obj.get_ECI(self.time/86400)[0:3]
        range = np.sqrt((x-x_s)**2+(y-y_s)**2+(z-z_s)**2)
        return range

        
class RangeRateMeas(Meas):
    ''' Range Rate Measurement
    
    Includes information common to all measurements plus addition info (station_obj) specific to
    Range Rate measurements
    
    Arguments:
        - station_obj: (station) Station object from Amazon groundstation (may be expanded in the
        future) which includes latitude, longitude of station
    '''
    def __init__(self,value,time,mission_id,station_obj,sigma=0.001):
        super(RangeRateMeas,self).__init__(value,time,mission_id,sigma)
        self.station_obj = station_obj
    
    def __repr__(self):
        return "Range Rate is {} km/s from {} Station".format(self.value, self.station_obj.name)  
     
    def gen_meas(self,state):
        x,y,z,vx,vy,vz = state
        x_s,y_s,z_s,vx_s,vy_s,vz_s = self.station_obj.get_ECI(self.time/86400)
        range = np.sqrt((x-x_s)**2+(y-y_s)**2+(z-z_s)**2)
        range_rate = ((x-x_s)*(vx-vx_s)+(y-y_s)*(vy-vy_s)+(z-z_s)*(vz-vz_s))/range
        return range_rate
    
        
class ElevMeas(Meas):
    ''' Elevation Measurement
    
    Includes information common to all measurements plus addition info (station_obj) specific to
    Elevation measurements
    
    Arguments:
        - station_obj: (station) Station object from Amazon groundstation (may be expanded in the
        future) which includes latitude, longitude of station
    '''
    def __init__(self,value,time,mission_id,station_obj,sigma=1):
        super(ElevMeas,self).__init__(value,time,mission_id,sigma)
        self.station_obj = station_obj
        
    def __repr__(self):
        return "Elevation is {} degrees from {} Station".format(self.value, self.station_obj.name)
      
    def gen_meas(self,state):
        x,y,z = state[0:3]
        x_s,y_s,z_s = self.station_obj.get_ECI(self.time/86400)[0:3]
        xy_range = np.sqrt((x-x_s)**2+(y-y_s)**2)
        z_range = z-z_s
        elev = atan(z_range/xy_range)*180/np.pi
        return elev
    
        
class Station:
    ''' Station object
    Represents a groundstation. Inputs are latitude and longitude of the station, but the convert
    method converts to earth centered inertial  coordinates.
    
    Arguments:
        latitude: (float64) latitude of the ground station (in degrees)
        longitude: (float64) longitude of the ground station (in degrees)
        time: (float64) Modified Julian Date
        name: (string) Station Name
        altitude (optional): (float64) altitude of ground station (in feet)
    '''
    def __init__(self,name,latitude,longitude,height=None):
        self.latitude = latitude
        self.longitude = longitude
        self.time = time
        self.name = name
        if height:
            self.height = WMS.r_Ea + height/3280.8399167
        else:
            self.height = WMS.r_Ea
            
    def __repr__(self):
        val = "{0} Station at {1} degrees latitude, {2} degrees longitude, {3:.2f} km altitude"
        return val.format(self.name, self.latitude, self.longitude, self.height)
        
    def get_ECI(self, t):
        date = time.Time(t, format='mjd')
        pos, vel = astro.EarthLocation.from_geodetic(self.longitude, self.latitude, self.height).get_gcrs_posvel(date)
        ECEF = np.append(pos.xyz.to(u.km).value, vel.xyz.to(u.km/u.s).value)
        return ECEF
    
    
