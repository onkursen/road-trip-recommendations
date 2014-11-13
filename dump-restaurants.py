# Get all restaurants from a desired city (in this case, Phoenix)
# and save them to a file using Pickle, Python's persistence model.

from util import *

# Use logging file configuration
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('dump_restaurants')

BUSINESS_PATH = DATASET_PATH + 'yelp_academic_dataset_business.json'

restaurants = []
file_businesses = open(BUSINESS_PATH)
for line in file_businesses:
    if line.strip() == '':
        continue
    
    # Python json module doesn't play nice with newlines
    # (which show up in addresses)
    line_with_newlines_removed = re.sub('\n', ', ', line)

    # Only select restaurants from the desired city
    business = json.loads(line)
    logger.debug("Processing business: %s" % business['name'])
    if business['city'] == DESIRED_CITY and \
       'Restaurants' in business['categories']:
        logger.debug('Business is a restaurant in %s' % DESIRED_CITY)
        restaurants.append(business)
file_businesses.close()

logger.debug('%d restaurants in %s found' % (len(restaurants), DESIRED_CITY))

# Save restaurants to the output using Pickle, Python's
# persistence model
logger.debug('Saving restaurants to file: %s' % PICKLE_PATH)
joblib.dump(restaurants, PICKLE_PATH)

