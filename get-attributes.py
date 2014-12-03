# Get all restaurants from a desired city (in this case, Phoenix)
# and save them to a file using Pickle, Python's persistence model.

from util import *

# Use logging file configuration
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('dump_restaurants')

BUSINESS_PATH = DATASET_PATH + 'yelp_academic_dataset_business.json'

attributes = set()
file_businesses = open(BUSINESS_PATH)
for line in file_businesses:
    if line.strip() == '':
        continue
    
    # Python json module doesn't play nice with newlines
    # (which show up in addresses)
    line_with_newlines_removed = re.sub('\n', ', ', line)

    # Only select restaurants from the desired city
    business = json.loads(line)
    if business['city'] == DESIRED_CITY and \
       'Restaurants' in business['categories']:
        business_attributes = business['attributes']
        if 'stars' not in business:
            logger.debug("Processing business: %s" % business['name'])
            print 'PROBLEM'
        for attr in business_attributes:
            if type(business_attributes[attr]) == bool:
                attributes.add(attr)
file_businesses.close()

attributes.add('Price Range')
logger.debug('%d attributes in %s found' % (len(attributes), DESIRED_CITY))
for attr in attributes:
    print attr

# Save restaurants to the output using Pickle, Python's
# persistence model
logger.debug('Saving restaurants to file: %s' % ATTRIBUTES_PATH)
joblib.dump(attributes, ATTRIBUTES_PATH )

