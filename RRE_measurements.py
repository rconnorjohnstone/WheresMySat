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
from measurements import Meas
import astropy.coordinates as astro
from astropy import units as u
from astropy import time


class RangeMeas(Meas):
    ''' Range Measurement
    
    Includes information common to all measurements plus addition info (station_id) specific to
    Range measurements
    
    Arguments:
        - station_id: (station) Station object from Amazon groundstation (may be expanded in the
        future) which includes latitude, longitude of station
    '''
    def __init__(self,value,time,mission_id,station_id):
        self.range_meas = value
        super(RangeMeas,self).__init__(value,time,mission_id)
        self.station_id = station_id
        
    def __repr__(self):
        return "Range is {} km".format(self.range_meas)
        
class RangeRateMeas(Meas):
    ''' Range Rate Measurement
    
    Includes information common to all measurements plus addition info (station_id) specific to
    Range Rate measurements
    
    Arguments:
        - station_id: (station) Station object from Amazon groundstation (may be expanded in the
        future) which includes latitude, longitude of station
    '''
    def __init__(self,value,time,mission_id,station_id):
        self.rangerate = value
        super(RangeRateMeas,self).__init__(value,time,mission_id)
        self.station_id = station_id
        
    def __repr__(self):
        return "Range Rate is {} km/s".format(self.rangerate)
        
class ElevMeas(Meas):
    ''' Elevation Measurement
    
    Includes information common to all measurements plus addition info (station_id) specific to
    Elevation measurements
    
    Arguments:
        - station_id: (station) Station object from Amazon groundstation (may be expanded in the
        future) which includes latitude, longitude of station
    '''
    def __init__(self,value,time,mission_id,station_id):
        self.elev = value
        super(ElevMeas,self).__init__(value,time,mission_id)
        self.station_id = station_id
        
    def __repr__(self):
        return "Elevation is {} degrees".format(self.elev)
        
class station:
    ''' Station object
    Represents a groundstation. Inputs are latitude and longitude of the station, but the convert
    method converts to earth centered inertial  coordinates.
    
    Arguments:
        latitude: (float64) latitude of the ground station (in degrees)
        longitude: (float64) longitude of the ground station (in degrees)
        time: (float64) Modified Julian Date
        altitude (optional): (float64) altitude of ground station (in feet)
    '''
    def __init__(self,latitude,longitude,time,*args):
        self.latitude = latitude
        self.longitude = longitude
        self.time = time
        if len(args) == 1: self.height = WMS.r_Ea + args[0]*0.0003048
        else: self.height = WMS.r_Ea
        
    def get_ECI(self):
        date = time.Time(self.time, format='mjd')
        pos, vel = astro.EarthLocation.from_geodetic(self.longitude, self.latitude, self.height).get_gcrs_posvel(date)
        ECEF = np.append(pos.xyz.to(u.km).value, vel.xyz.to(u.km/u.s).value)
        return ECEF
    
    def __repr__(self):
        return "Station at {}".format(self.get_ECI())