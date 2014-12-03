import json
import logging
import logging.config
from math import radians, cos, sin, asin, sqrt, exp
from os import path
from random import random
import re
from sklearn import linear_model
from sklearn.externals import joblib

DESIRED_CITY = 'Phoenix'
RESTAURANTS_PATH = 'restaurants_%s.pkl' % (DESIRED_CITY.lower())
ATTRIBUTES_PATH = 'attributes_%s.pkl' % (DESIRED_CITY.lower())
DATASET_PATH = '/Users/mukul/Downloads/yelp_dataset/' # Change this for your machine

# Taken from: http://stackoverflow.com/questions/4913349/
# Returns the distance between a pair of lat/longs
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km
