# Loads preprocessed data and does a linear regression using features:
# 1. (Distance from start to restaurant) + (Distance from restaurant to end) - (Distance from start to end)
# 2. Rating

from util import *

# Use logging file configuration
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('regression')

print 'Loading saved restaurants'
restaurants = joblib.load(PICKLE_PATH)

print 'Limiting restaurants'
LIMIT_RESTAURANTS = True
NUM_RESTURANTS = 100 * LIMIT_RESTAURANTS + len(restaurants) * (not LIMIT_RESTAURANTS)
restaurants = restaurants[:NUM_RESTURANTS]

# longitude then latitude
START = (-111.995428, 33.451096)
END   = (-112.117308, 33.423449)
distance_from_start_to_end = haversine(START[0], START[1], END[0], END[1])

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

	features[i] = (
		added_distance, 
		b['stars']		# star rating
	)
	if added_distance < 0:
		print features[i]

# Distance metrics
DISCOUNT_FACTOR = 10
BASE = 1.3
def exp_distance_metric(d):
	return exp( (-1.0 * d ** 2) / (2 * DISCOUNT_FACTOR ** 2) )

def hyperbolic_distance_metric(d):
	return 1.0 / (1 + DISCOUNT_FACTOR * d)

def linear_rating_metric(r):
	return r

def exp_rating_metric(r):
	return BASE ** r

def score(featureVector):
	added_distance, rating = featureVector
	return exp_rating_metric(rating) * exp_distance_metric(added_distance)

print 'Calculating scores and establishing datasets'
scores = map(score, features)
training_set = restaurants[:NUM_RESTURANTS / 2]
training_set_features = features[:NUM_RESTURANTS / 2]
training_set_scores = scores[:NUM_RESTURANTS / 2]

test_set = restaurants[NUM_RESTURANTS / 2:]
test_set_features = features[NUM_RESTURANTS / 2:]
test_set_scores = scores[NUM_RESTURANTS / 2:]

print 'Baseline: pick restaurant with lowest distance and highest rating'
lowest_distance = min(test_set_features)[0]
closest_restaurants = [test_set[i] for i in range(NUM_RESTURANTS / 2) if test_set_features[i][0] == lowest_distance]
best_restaurant = max(closest_restaurants, key=lambda restaurant: restaurant['stars'])
print 'Baseline recommends:'
print best_restaurant['name'], best_restaurant['full_address'], best_restaurant['longitude'], best_restaurant['latitude']
print 'This adds a distance of %f km away with a %.1f-star rating' % (lowest_distance, best_restaurant['stars'])
print

print 'Running regression against training set'
regr = linear_model.LinearRegression()
regr.fit(training_set_features, training_set_scores)

def percent_error(true_value, observed_value):
	return abs(true_value - observed_value) / true_value * 100.0

print 'Predicting test set'
predictions = regr.predict(test_set_features)
errors = []
print 'Case\tPredicted\tActual\tPercent error'
for i, score in enumerate(predictions):
	print '%-2d\t\t%-.3f\t%.3f\t%.1f' % (i+1, score, scores[i], percent_error(scores[i], score))
	errors.append(percent_error(scores[i], score))
print 'Average percent error', sum(errors) / len(errors)

recommended_index = max((score, i) for i, score in enumerate(predictions))[1]
recommended = test_set[recommended_index]
print 'Regression recommends:'
print recommended['name'], recommended['full_address'], recommended['longitude'], recommended['latitude']
print 'This adds a distance of %f km away with a %.1f-star rating' % (test_set_features[recommended_index][0], recommended['stars'])

