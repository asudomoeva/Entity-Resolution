
# coding: utf-8

# pip install affinegap
# pip install python-levenshtein

import json
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as prf_score
from sklearn.metrics import accuracy_score as accuracy_score
import Levenshtein as lv
from difflib import SequenceMatcher
import affinegap
import scipy
import timeit
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint as sp_randint

def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):
    """
        In this function, You need to design your own algorithm or model to find the matches and generate
        a matches_test.csv in the current folder.

        you are given locu_train, foursquare_train json file path and matches_train.csv path to train
        your model or algorithm.

        Then you should test your model or algorithm with locu_test and foursquare_test json file.
        Make sure that you write the test matches to a file in the same directory called matches_test.csv.

    """
    #define clean function for foursquare datasets (train & test)
    def clean_foursquare(dataset):
        #OVERALL (NaN and None)
        dataset.replace('', np.nan, inplace=True)
        dataset.fillna(value=np.nan, inplace=True)
        # Dropping country and region (non-original)
        dataset.drop(['country','region'],axis=1, inplace=True)

        #FEATURE: PHONE 
        #align with locu formatting
        dataset['phone']=dataset['phone'].str[1:4]+dataset['phone'].str[6:9]+dataset['phone'].str[10:14]
        #we dont know if there will be any nulls in the hidden test, check for all possible nulls to avoid null matching

        #FEATURE: WEBSITE
        dataset['website'] = dataset['website'].str.split('.com').str[0]
        #I know there has to be a better way to do this, but the replace function with lists does not seem to be working:(
        dataset['website'] = dataset['website'].str.replace('http:','').str.replace(' ', '').str.replace('.us', '').str.replace('.geomerx', '').str.replace('.org', '').str.replace('.blogspot', '').str.replace('.tumblr', '').str.replace('.net', '').str.replace('https:', '').str.replace('www.', '').str.replace('/', '')
        dataset['website'] = dataset['website'].str.upper()

        #FEATURE: STREET_ADDRESS
        dataset['street_address'] = dataset['street_address'].str.upper()
        dataset['street_address'] = dataset['street_address'].str.split(' #').str[0]
        dataset['street_address'] = dataset['street_address'].str.split(',').str[0]
        dataset['street_address'] = dataset['street_address'].str.replace(' ', '').str.replace('.', '').str.replace('STREET', "ST").str.replace('AVENUE', 'AVE').str.replace('BOULEVARD','BLVD').str.replace('PLAZA', 'PLZ').str.replace('!', '').str.replace('SQUARE', 'SQ').str.replace('PLACE', 'PL').str.replace('WEST ', 'W ').str.replace('EAST ', 'E ')

        #FEATURE: NAME
        dataset['name'] = dataset['name'].str.upper()
        dataset['name'] = dataset['name'].str.replace('\'', '').str.replace(' ', '').str.replace('É', 'E').str.replace('&', 'AND').str.replace('-', '').str.replace('\(', '').str.replace('\)', '').str.replace('/', '')

        #FEATURE: LOCALITY
        dataset['locality'] = dataset['locality'].str.upper()

        return dataset

    #define clean function for locu datasets (train & test)

    def clean_locu(dataset):
        #OVERALL (NaN and None)
        dataset.replace('', np.nan, inplace=True)
        dataset.fillna(value=np.nan, inplace=True)
        # Dropping country and region (non-original)
        dataset.drop(['country','region'],axis=1, inplace=True)

        #FEATURE:PHONE
        dataset['phone'] = dataset['phone'].str.replace('x', '')

        #FEATURE: WEBSITE
        dataset['website'] = dataset['website'].str.split('.com').str[0]
        #I know there has to be a better way to do this, but the replace function with lists does not seem to be working:(
        dataset['website'] = dataset['website'].str.replace('http:','').str.replace(' ', '').str.replace('.us', '').str.replace('.geomerx', '').str.replace('.org', '').str.replace('.blogspot', '').str.replace('.tumblr', '').str.replace('.net', '').str.replace('https:', '').str.replace('www.', '').str.replace('/', '')
        dataset['website'] = dataset['website'].str.upper()

        #FEATURE: STREET ADDRESS
        dataset['street_address'] = dataset['street_address'].str.upper()
        dataset['street_address'] = dataset['street_address'].str.split(' #').str[0]
        dataset['street_address'] = dataset['street_address'].str.split(',').str[0]
        dataset['street_address'] = dataset['street_address'].str.replace('.', '').str.replace(' ', '').str.replace('STREET', "ST").str.replace('AVENUE', 'AVE').str.replace('BOULEVARD','BLVD').str.replace('PLAZA', 'PLZ').str.replace('!', '').str.replace('SQUARE', 'SQ').str.replace('PLACE', 'PL').str.replace('WEST ', 'W ').str.replace('EAST ', 'E ')

        #FEATURE: NAME
        # SHOOULD WE ALIGN THINGS LIKE PIZZERIA VS PIZZA AS PART OF DATA CLEANING?
        dataset['name'] = dataset['name'].str.upper()
        dataset['name'] = dataset['name'].str.replace('\'', '').str.replace(' ', '').str.replace('É', 'E').str.replace('&', 'AND').str.replace('-', '').str.replace('\(', '').str.replace('\)', '').str.replace('/', '')

        #FEATURE: LOCALITY
        dataset['locality'] = dataset['locality'].str.upper()

        return dataset

    #foursquare_train_path = 
    #Perform Cleaning on the given datasets
    foursquare_train = clean_foursquare(foursquare_train_path)
    locu_train = clean_locu(locu_train_path)
    foursquare_test = clean_foursquare(foursquare_test_path)
    locu_test = clean_locu(locu_test_path)

    #create the new dataframe
    def create_dataframe(foursquare, locu):
        foursquare_ids_list = list(foursquare['id'])
        locu_ids_list = list(locu['id'])
        length_fids = len(foursquare_ids_list)
        length_lids = len(locu_ids_list)

        locu_ids_repeated = np.repeat(locu_ids_list,length_fids)
        foursquare_ids_tiled = np.tile(foursquare_ids_list,length_lids)

        df = pd.DataFrame({'locu_id': locu_ids_repeated,'foursquare_id':foursquare_ids_tiled})
        foursquare = foursquare.add_suffix('_F')
        locu = locu.add_suffix('_L')
        df = df.merge(foursquare,left_on='foursquare_id',right_on=['id_F'],how='left').merge(locu,left_on='locu_id',right_on='id_L', how='left')
        df['unique_id'] = df['foursquare_id'] + df['locu_id']
        return df


    #apply the function on train and test
    df_train = create_dataframe(foursquare_train, locu_train)
    df_test = create_dataframe(foursquare_test, locu_test)

    # metrics for distance between strings
    def either_string_is_null(str1,str2):
        if pd.isnull(str1) or pd.isnull(str2):
            return True
        else:
            return False

    def aff(str1,str2):
        if either_string_is_null(str1,str2):
            return np.nan
        else:
            return affinegap.affineGapDistance(str1,str2)

    def lev(str1,str2):
        if either_string_is_null(str1,str2):
            return np.nan
        else:
            return lv.distance(str1, str2)

    def sim(str1, str2):
        if either_string_is_null(str1,str2):
            return np.nan
        else:
            return SequenceMatcher(None, str1, str2).ratio()

    def lenlongcommon(str1,str2):
        if either_string_is_null(str1,str2):
            return np.nan
        else:
        # initialize SequenceMatcher object with 
         # input string
            seqMatch = SequenceMatcher(None,str1,str2)

             # find match of longest sub-string
             # output will be like Match(a=0, b=0, size=5)
            match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))
        return match.size

    def add_features(df):
        #FEATURE: PHONE
        #perfect match
        df['phone_perfect_match'] = df['phone_F'] == df['phone_L']

        # Phone number filled in, but not a match
        df['phone_filled_in'] = df['phone_F'].notnull() & df['phone_L'].notnull()
        df['phone_different_both_filled_in'] = (df['phone_perfect_match']==False) & (df['phone_filled_in'])

        #FEATURE: LATITUDE
        #perfect match
        df['latitude_perfect_match'] = df['latitude_F'] == df['latitude_L']
        #Distance between
        df['latitude_diff'] = df['latitude_F'] - df['latitude_L']

        #FEATURE: LONGITUDE
        #perfect match
        df['longitude_perfect_match'] = df['longitude_F'] == df['longitude_L']
        #Distance between
        df['longitude_diff'] = df['longitude_F'] - df['longitude_L']

        #LATLONG (both match)
        df['lat_and_long_match'] = df['latitude_perfect_match'] & df['longitude_perfect_match']

        #df['lat_and_long_filled_in'] = df['latitude_F'].notnull() & df['latitude_L'].notnull() & df['longitude_F'].notnull() & df['longitude_L'].notnull()
        #df['lat_and_long_different_both_filled_in'] = (df['lat_and_long_match']==False) & (df['lat_and_long_filled_in'])

        #df['latitude_within_0005'] = df['latitude_diff']<.0005 & df['latitude_diff']>(-0.0005)
        #df['longitude_within_0005'] = df['longitude_diff']<.0005 & df['longitude_diff']>(-0.0005)

        #FEATURE: NAME
        #perfect match
        df['name_perfect_match'] = df['name_F'] == df['name_L']

        #FEATURE: WEBSITE
        #perfect match
        df['website_perfect_match'] = df['website_F'] == df['website_L']

        #FEATURE: ADDRESS
        #perfect match
        df['address_perfect_match'] = df['street_address_F'] == df['street_address_L']

        #FEATURE: POSTAL_CODE
        #perfect mismatch
        df['postal_not_match'] = df['postal_code_F'] != df['postal_code_L']

        a = timeit.default_timer()

        for col in ['name','street_address','website']:
            print('aff,lev,sim,llc')
            print(col)
            df[col+'_aff'] = df[[col+'_F',col+'_L']].apply(lambda x: aff(*x), axis=1)
            #Your statements here

            b = timeit.default_timer()
            print(b-a)

            df[col+'_lev'] = df[[col+'_F',col+'_L']].apply(lambda x: lev(*x), axis=1)
            c = timeit.default_timer()
            print(c-b)

            df[col+'_sim'] = df[[col+'_F',col+'_L']].apply(lambda x: sim(*x), axis=1)
            d = timeit.default_timer()
            print(d-c)

            df[col+'_llc'] = df[[col+'_F',col+'_L']].apply(lambda x: lenlongcommon(*x), axis=1)
            e = timeit.default_timer()
            print(e-d)

        return df

    #add features to the data frames
    df_train_with_created_features = add_features(df_train)
    df_test_with_created_features = add_features(df_test)


    #Add matches column
    matches_train = matches_train_path.rename(index=str,columns={'foursquare_id':'true_foursquare_id'})
    df_train_with_created_features_with_FID = df_train_with_created_features.merge(matches_train, on='locu_id', how='outer')

    #add a target
    df_train_with_created_features_with_FID['target'] = df_train_with_created_features_with_FID['foursquare_id']==df_train_with_created_features_with_FID['true_foursquare_id']

    features_to_keep = ['phone_perfect_match', 'phone_different_both_filled_in', 'lat_and_long_match', 'latitude_diff', 'longitude_diff', 'name_perfect_match', 'website_perfect_match', 'address_perfect_match', 'postal_not_match']+[col for col in df_train_with_created_features_with_FID.columns if 'aff' in col or 'lev' in col or 'sim' in col or 'llc' in col]

    #clean train data
    X_train = df_train_with_created_features_with_FID[features_to_keep]
    id_mapping_train = df_train_with_created_features_with_FID[['foursquare_id', 'locu_id']]
    # clean test data
    X_test = df_test_with_created_features[features_to_keep]
    id_mapping_test = df_test_with_created_features[['foursquare_id', 'locu_id']]
    #Create target
    target = df_train_with_created_features_with_FID['target']

    #deal with NaN to pass into the random forest
    X_train = X_train.fillna(-10000, axis =1).astype(int)
    X_test = X_test.fillna(-10000, axis =1).astype(int)

    import sklearn
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    from sklearn.feature_selection import SelectFromModel
    clf1 = RandomForestClassifier()

    # Set a minimum threshold of 0.25
    sfm = SelectFromModel(clf1, threshold='.15*mean')
    sfm.fit(X_train, target)
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)
    param_grid = {"max_depth": [None],
                  "max_features": ['log2','sqrt'],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                 "class_weight": [None]}
    grid = GridSearchCV(RandomForestClassifier(random_state=0),param_grid=param_grid, cv=StratifiedKFold())
    grid.fit(X_train, target)
    print(grid.cv_results_)

    predict_train = grid.predict_proba(X_train)
    predict_test = grid.predict_proba(X_test)
    predict1 = pd.DataFrame(predict_train)
    predict1_test = pd.DataFrame(predict_test)


    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    X_train['predict_proba'] = predict1[1]
    X_test['predict_proba'] = predict1_test[1]

    # add id mapping
    X_train_with_id = pd.concat([X_train,id_mapping_train],axis=1)
    X_test_with_id = pd.concat([X_test,id_mapping_test],axis=1)

    #add predicted probability
    X_train_with_id['y_pred'] = (X_train_with_id['predict_proba']>.3).astype(int)
    X_test_with_id['y_pred'] = (X_test_with_id['predict_proba']>.3).astype(int)

    #create the final test output
    final_test_output = X_test_with_id[X_test_with_id['y_pred']==1][['locu_id', 'foursquare_id']]
    
    final_test_output.to_csv('matches_test.csv', index = False)
    
    pass

