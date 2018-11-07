# Entity Resolution

## Problem Description 
Given two datasets that describe the some of the same entities, we need to identify whether an entity in one dataset is the same as an entity in the other dataset. Our datasets were provided by Foursquare and Locu, and contain descriptive information about various venues such as venue names and phone numbers.

## Data
The data and labels for training are in 'train' folder including the following:
foursquare_train.json
locu_train.json
matches_train.csv

The data for test validation is 'test' in folder including the following:
foursquare_test.json
locu_test.json

The 'json' files contain a json-encoded list of venue attribute dictionaries. The 'csv' file contains two columns, 'locu_id' and 'foursquare_id', which reference the venue 'id' fields that match in each dataset.

The two datasets don't have the same exact formatting for some fields: check out the 'phone' field in each dataset as an example. Therefore, some of your datasets had to be normalized during preprocessing.

## Final Outcome

The main goal of the project is to generate 'matches_test.csv', a mapping that looks like 'matches_train.csv' but with mappings for the new test listings with very high accuracy.


## Performance Measure

To assess the model, we will be measuring the precision, recall, and F1-score of the algorithm against the ground truth in "matches_train.csv". 
