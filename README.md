# Road Trip Recommendations
Using the [Yelp Academic Dataset](https://www.yelp.com/academic_dataset),
Python, and scikit-learn,
I worked with Jocelyn Hickhox to implement an algorithm that provides
restaurant recommendations for someone on a road trip
given an origin, a destination, and various desired attributes.
This was done as part of a project for CS 221 (Artificial Intelligence) at
Stanford University.

Released under the MIT license (see `LICENSE`).

## Examples

```
Loading saved restaurants
Limiting restaurants
Constructing features
Calculating scores and establishing datasets
Baseline: pick restaurant with lowest distance and highest rating
Baseline recommends:
Pete's Fish & Chips 3715 E Van Buren St
Phoenix, AZ 85008 -112.001533 33.45111
This adds a distance of 0.021151 km away with a 3.0-star rating
```

```
Running regression against training set
Predicting test set
Average percent error 20.8014599346
Regression recommends:
Pozoleria Guerrero 2801 E Van Buren St
Phoenix, AZ 85008 -112.021191 33.451004
This adds a distance of 0.100828 km away with a 4.5-star rating
```
