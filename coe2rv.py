# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:57:34 2019

@author: chels
"""
from numpy import sin,cos
import numpy as np

def rad2deg(ang):
    ang = ang*180./np.pi
    return ang

def deg2rad(ang):
    ang = ang*np.pi/180.
    return ang

def coe2rvfunc(a,e,i,node,w,v):
    p=a*(1.-e**2)
    mu = 398600.4418#for km
    r = np.array([p*cos(v)/(1+e*cos(v)), p*sin(v)/(1+e*cos(v)), 0.]).reshape((3,1))
    rd = np.array([-np.sqrt(mu/p)*sin(v),np.sqrt(mu/p)*(e+cos(v)), 0.]).reshape((3,1))
    rmat = np.array([[cos(node)*cos(w)-sin(node)*sin(w)*cos(i), \
                      -cos(node)*sin(w)-sin(node)*cos(w)*cos(i),\
                      sin(node)*sin(i)],\
    [sin(node)*cos(w)+cos(node)*sin(w)*cos(i),\
     -sin(node)*sin(w)+cos(node)*cos(w)*cos(i),\
     -cos(node)*sin(i)],\
     [sin(w)*sin(i), cos(w)*sin(i), cos(i)]])
    r = rmat @ r
    rd = rmat @ rd
    state = np.concatenate((r,rd))
    return state
