# Loads preprocessed data and does a linear regression using features:
# 1. (Distance from start to restaurant) + (Distance from restaurant to end) - (Distance from start to end)
# 2. Rating

from util import *
from sys import argv
from collections import Counter

# Use logging file configuration
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('regression')

print 'Loading saved restaurants and attributes'
restaurants = joblib.load(RESTAURANTS_PATH)
attributes = joblib.load(ATTRIBUTES_PATH)

# Used for generating values for feature vector
attribute_map = {attr: i for i, attr in enumerate(attributes)}

print 'Limiting restaurants'
LIMIT_RESTAURANTS = False
NUM_RESTURANTS = 1000 if LIMIT_RESTAURANTS else len(restaurants)
restaurants = restaurants[:NUM_RESTURANTS]

ratings = Counter()
for r in restaurants:
	ratings[r['stars']] += 1
print 'Rating distribution', sorted(ratings.items())
print

# longitude then latitude
START = (-112.007616, 33.437278) # PHX
END   = (-112.070611, 33.449781) # Phoenix Convention Center
distance_from_start_to_end = haversine(START[0], START[1], END[0], END[1])

print 'Processing desired attributes'
try:
	desired_attributes = open(argv[1]).read().split('\n')
except:
	print 'You need an attribute file!'
	exit(1)

print 'Constructing features'
features = [None] * len(restaurants)
for i, b in enumerate(restaurants):
	lon, lat = b['longitude'], b['latitude']

	# additional distance by taking stop
	distance_from_start = haversine(lon, lat, *START)
	distance_to_end = haversine(lon, lat, *END)
	added_distance = distance_from_start + \
					 distance_to_end - \
					 distance_from_start_to_end

	# Attribute values represented as an array of floats
	b_attributes = [0] * len(restaurants)
	for attr in b['attributes']:
		if attr in attribute_map:
			b_attributes[attribute_map[attr]] = float(b['attributes'][attr])

	features[i] = (
		added_distance,
		b['stars']		# star rating
	) + tuple(b_attributes)

# Distance metrics
DISCOUNT_FACTOR = 10
def gaussian_distance_metric(d):
	return exp( (-1.0 * d ** 2) / (2 * DISCOUNT_FACTOR ** 2) )

def hyperbolic_distance_metric(d):
	return 1.0 / (1 + DISCOUNT_FACTOR * d)

def desired_attributes_metric(attributes, rating):
	result = 1
	# Price Range and stars are special case
	for f in desired_attributes:
		# Check if restaurant in desired price range
		if f[:11] == 'Price Range':
			result += attributes[attribute_map['Price Range']] in eval(f[11:])
		# Check if restaurant meets minimum star requirement
		elif f[:5] == 'Stars':
			result += rating <= float(f[5:])
		# Check if boolean value is met
		else:
			result += f in attribute_map and attributes[attribute_map[f]] != 0
	return result

# Rating metrics
BASE = 1.3
def linear_rating_metric(r):
	return r

def exp_rating_metric(r):
	return BASE ** r

# Score multiples rating, distances, and desired_attributes metrics
def score(featureVector):
	added_distance, rating, attributes = featureVector[0], featureVector[1], featureVector[2:]
	return exp_rating_metric(rating) * gaussian_distance_metric(added_distance) * \
	desired_attributes_metric(attributes, rating)

print 'Calculating scores and establishing datasets'
scores = map(score, features)
training_set = restaurants[:NUM_RESTURANTS / 2]
training_set_features = features[:NUM_RESTURANTS / 2]
training_set_scores = scores[:NUM_RESTURANTS / 2]

test_set = restaurants[NUM_RESTURANTS / 2:]
test_set_features = features[NUM_RESTURANTS / 2:]
test_set_scores = scores[NUM_RESTURANTS / 2:]

print
print 'Baseline: pick restaurant with lowest distance and highest rating'
lowest_distance = min(test_set_features)[0]
closest_restaurants = [
	test_set[i]
	for i in range(NUM_RESTURANTS / 2)
	if test_set_features[i][0] == lowest_distance
]
best_restaurant = max(closest_restaurants, key=lambda restaurant: restaurant['stars'])
print 'Baseline recommends:'
print best_restaurant['name'], best_restaurant['full_address'], \
best_restaurant['longitude'], best_restaurant['latitude']
print 'This adds a distance of %f km away with a %.1f-star rating and a %d price range' % (
	lowest_distance,
	best_restaurant['stars'],
	best_restaurant['attributes']['Price Range']
)
print

print 'Running regression against training set'
regr = linear_model.LinearRegression()
regr.fit(training_set_features, training_set_scores)

def percent_error(true_value, observed_value):
	return abs(true_value - observed_value) / true_value * 100.0

print 'Predicting test set'
predictions = regr.predict(test_set_features)
errors = []
for i, score in enumerate(predictions):
	errors.append(percent_error(scores[i], score))
print 'Average percent error', sum(errors) / len(errors)
print

print 'Start point', START
print 'End point', END

recommended_index = max((score, i) for i, score in enumerate(predictions))[1]
recommended = test_set[recommended_index]
print 'Regression recommends:'
print recommended['name'], recommended['full_address'], \
recommended['longitude'], recommended['latitude']
print 'This adds a distance of %f km with a %.1f-star rating and a %d price range' % (
	test_set_features[recommended_index][0],
	recommended['stars'],
	recommended['attributes']['Price Range']
)
