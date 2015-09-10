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
RESTAURANTS_PATH = 'data_dumps/restaurants_%s.pkl' % (DESIRED_CITY.lower())
ATTRIBUTES_PATH = 'data_dumps/attributes_%s.pkl' % (DESIRED_CITY.lower())
DATASET_PATH = '/Users/mukul/Downloads/yelp_dataset/' # Change this for your machine

RADIUS_OF_EARTH = 6367 # km

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

  return RADIUS_OF_EARTH * c
