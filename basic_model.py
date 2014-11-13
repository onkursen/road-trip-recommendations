from util import *
regr = linear_model.LinearRegression()

# Small dataset for intiial testing
# Feature vectors contain [distance, average rating]
restaurants = [[5.0, 3.5], [9.4, 4.0], [5.1, 4.0], [12.5, 4.8], [1.2, 2.0], [7.4, 4.1], [3.0, 4.0], [2.9, 3.9], [6.8, 4.3], [7.0, 3.14]] 
devData = [[6.0, 4.0], [2.1, 3.0], [1.6, 2.8], [6.7, 4.8], [4.1, 4.1]]

# very simple formula to generate rough target values
def getTargetValues(trainingData):
	values = []
	for distance, rating in trainingData:
		value = (rating**2) / distance 
		values.append(value)
	return values

regr.fit(restaurants, getTargetValues(restaurants))
predictions = regr.predict(devData)
print "--------Begin Simple Test-----------"
print "\n"
print "--------Fit to data----------"
print "-----restaurant data:"
print restaurants
print "-----Coefficients:"
print(regr.coef_)
print "-------Predict new scores-------"
print "----(distance, rating), score:"
for i, val in enumerate(predictions):
	print devData[i], val
