from util import *

NUM_RESTAURANTS = 50
SELECTED_OUTPUT_PATH = 'selected_restaurants.pkl'

if not path.isfile(PICKLE_PATH):
	raise Exception("Dump of all restaurants doesn't exist yet! Run dump_restaurants.py first.")

all_restaurants = joblib.load(PICKLE_PATH)
selected_restaurants = all_restaurants[:NUM_RESTAURANTS]

joblib.dump(selected_restaurants, SELECTED_OUTPUT_PATH)
