#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
import logging

from past_features import *
from elo_features import *
from categorical_features import *
from strategy_assessment import *
from utilities import *
from visualize import *
import glob
from clean import prepare_tennis_data
from config import *
from config import get_xgb_params
######################### Building of the raw dataset ##########################
# Some preprocessing is necessary because for several years the odds are not present
# We consider only the odds of Pinnacle. To do: remove B365 odds from the dataset
###################################################### RUN 1 ###################################################################
filenames=list(glob.glob(DATA_FILES_PATH))

l = [pd.read_excel(filename) for filename in filenames]

# prepare and clean the data:
data = prepare_tennis_data(l)


# print("2025 data:")
# print(data[data['Date'] > '2025-01-01'].tail())
# data_2025 = data[data['Date'] > '2025-01-01']
# print("removing 2025 data from the dataset")
# data = data[data['Date'] < '2025-01-01']  
# print("2025 data removed")
# print(data.tail())

# print("2025 first rounds:")
# print(data_2025[data_2025['Round'] == '1st Round'].tail())
# Check if data was successfully prepared
if data is None:
    logging.error("Failed to prepare tennis data. The input list 'l' is empty.")
    # You might want to handle this case, e.g., by loading default data or exiting
    # For example:
    # import sys
    # sys.exit("No data to process. Please check your data sources.")
else:
    ### Elo rankings data
    # Computing of the elo ranking of each player at the beginning of each match.
    match_ratings = compute_elo_ratings_fixed(data)
    features_df, labels = create_features_no_leakage(data, match_ratings)
    data = pd.concat([data, features_df.iloc[::2].reset_index(drop=True)], axis=1)  # Only add one row per match to original data
    # data = pd.concat([data,elo_rankings],axis=1)

    ### Storage of the raw dataset
    data.to_csv(ATP_DATA_PATH,index=False)


################################################################################
######################## Building training set #################################
################################################################################
### We'll add some features to the dataset
data=pd.read_csv(ATP_DATA_PATH)
# data.Date = data.Date.apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))
print(x)
data.Date = data.Date.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))


######################### The period that interests us #########################

beg = datetime.datetime(2005,1,1) 
# beg = datetime.datetime(2007,1,1) 
end = data.Date.iloc[-1]
indices = data[(data.Date>beg)&(data.Date<=end)].index

# # ################### Building of some features based on the past ################
# if REGENERATE_FEATURES:
    # features_player  = features_past_generation(features_player_creation,60,"playerft60",data,indices) #is this supposed to be 5?
    # dump(features_player,"player_features")
# features_player=load("player_features")

########################### Selection of our period ############################

data = data.iloc[indices,:].reset_index(drop=True)

########################## Encoding of categorical features ####################
# # features_categorical = data[["Tier","Court","Surface","Round","Best of","Tournament"]] # needed for WTA data?
# features_categorical = data[["Surface","Round","Tournament","Winner","Loser","Best of"]]
# features_categorical_encoded = categorical_features_encoding(features_categorical)
# players_encoded = features_players_encoding(data)
# tournaments_encoded = features_tournaments_encoding(data)
# features_onehot = pd.concat([features_categorical_encoded,
#                              players_encoded,
#                              tournaments_encoded
#                              ],axis=1)


############################### Duplication of rows ############################
## For the moment we have one row per match. 
## We "duplicate" each row to have one row for each outcome of each match. 
## Of course it isn't a simple duplication of each row, we need to "invert" some features

# Elo data
elo_rankings = data[["elo_a","elo_b","proba_elo"]]
elo_1 = elo_rankings
elo_2 = elo_1[["elo_b","elo_a","proba_elo"]]
elo_2.columns = ["elo_a","elo_b","proba_elo"]
elo_2.proba_elo = 1-elo_2.proba_elo
elo_2.index = range(1,2*len(elo_1),2)
elo_1.index = range(0,2*len(elo_1),2)
features_elo_ranking = pd.concat([elo_1,elo_2]).sort_index(kind='merge')

# Categorical features
# features_onehot = pd.DataFrame(np.repeat(features_onehot.values,2, axis=0),columns=features_onehot.columns)

# odds feature
odds = data[["PSW","PSL"]]

features_odds = pd.Series(odds.values.flatten(),name="odds")
features_odds = pd.DataFrame(features_odds)

# surface feature
features_surface = pd.get_dummies(data['Surface'],dtype=int,prefix="surface")
print(features_surface.tail())
# round feature
features_round = pd.get_dummies(data['Round'],dtype=int,prefix="round")
print(features_round.tail())

# After creating features_surface and features_round but before concatenation
features_surface = pd.concat([features_surface, features_surface]).reset_index(drop=True)
features_round = pd.concat([features_round, features_round]).reset_index(drop=True)

# Now all features should have 98388 rows

# to do:
# features_player

# features_tournament
features = pd.concat([
                features_odds,
                  features_elo_ranking,
                  features_surface,
                  features_round,
                #   features_winner,
                #   features_loser,
                  # comment out past features to see the effect on the ROI
                #   features_duo,   
                #   features_general,
                #   features_recent

                  ],axis=1, join='inner')



# After creating the final features DataFrame, convert all columns to numeric types
print("Converting features to numeric types...")

object_cols = features.select_dtypes(include=['object']).columns.tolist()
for col in object_cols:
    print(f"Converting obj column {col} from {features[col].dtype} to numeric")
    # Try to convert to numeric, fill NaN with 0
    features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)

# Before passing features to the model, check for and handle NaN values
nan_count = features.isna().sum().sum()
# if nan_count > 0:
#     print(f"Warning: Found {nan_count} NaN values in features. Filling with 0.")
#     features = features.fillna(0)
if features.isna().any().any():
    print(f"Warning: Found {features.isna().sum().sum()} NaN values in features. Filling with 0.")
    features = features.fillna(0)

check_feature_correlations(features, labels)
check_temporal_integrity(data, features)
features.to_csv(ATP_DATA_FEATURES_PATH,index=False)

################################################################################
#################### Strategy assessment - ROI computing #######################
################################################################################

## We adopt a sliding window method. We predict the outcome of delta consecutive matches , 
## with the N previous matches. A small subset of the training set is devoted to
## validation (the consecutive matches right before the testing matches)

######################### Confidence computing for each match ############################
features=pd.read_csv(ATP_DATA_FEATURES_PATH)


# start_date=datetime.datetime(2024,1,1) #first day of testing set
# print(start_date)
# test_beginning_match=data[data.Date==start_date].index[0] #id of the first match of the testing set
# span_matches=len(data)-test_beginning_match+1
# With this simpler approach:
# Total number of matches in the dataset
total_matches = len(data)
print(f"Total matches in dataset: {total_matches}")
duration_val_matches=300
duration_train_matches=10400
duration_test_matches=2000

# Calculate how many matches we need in total
required_matches = duration_train_matches + duration_val_matches + duration_test_matches
print(f"Required matches: {required_matches}")

# Check if we have enough data
if required_matches > total_matches:
    print(f"Warning: Not enough data. Adjusting durations.")
    # Adjust proportionally
    scale_factor = total_matches / required_matches
    duration_train_matches = int(duration_train_matches * scale_factor)
    duration_val_matches = int(duration_val_matches * scale_factor)
    duration_test_matches = total_matches - duration_train_matches - duration_val_matches
    print(f"Adjusted: Train={duration_train_matches}, Val={duration_val_matches}, Test={duration_test_matches}")

# Calculate the first test match index
test_beginning_match = total_matches - duration_test_matches
print(f"Test beginning match index: {test_beginning_match}")

# Calculate how many matches we have for testing
span_matches = duration_test_matches
print(f"Span of test matches: {span_matches}")

## Number of tournaments and players encoded directly in one-hot 
nb_players=50
nb_tournaments=5

# Load XGBoost parameters from config
xgb_params = get_xgb_params()
# Print the parameters to verify
print(f"XGBoost parameters from config: {xgb_params}")

# ## We predict the confidence in each outcome, "duration_test_matches" matches at each iteration
# key_matches=np.array([test_beginning_match+duration_test_matches*i for i in range(int(span_matches/duration_test_matches)+1)])
# Calculate key match starting points for testing
key_matches = []
current_match = test_beginning_match
while current_match < total_matches:
    key_matches.append(current_match)
    # Use a smaller step size if duration_test_matches is large
    step_size = min(duration_test_matches, 1000)  # Maximum 1000 matches per batch
    current_match += step_size

key_matches = np.array(key_matches)
print(f"Testing will be performed on {len(key_matches)} batches starting at indices: {key_matches}")

# Initialize the confs list
confs = []

# Loop through each test batch
for start in key_matches:
    print(f"\nProcessing batch starting at match {start}")
    
    # Calculate actual test duration for this batch (might be shorter for the last batch)
    actual_test_duration = min(duration_test_matches, total_matches - start)
    print(f"Using {duration_train_matches} training matches, {duration_val_matches} validation matches, and {actual_test_duration} test matches")
    
    # Call the simple_strategy function
    conf = simple_strategy(start, duration_train_matches, duration_val_matches, 
                          actual_test_duration, xgb_params, nb_players, 
                          nb_tournaments, features, data)
    
    # Debug output to see what's happening
    print(f"Returned conf type: {type(conf)}")
    if isinstance(conf, pd.DataFrame):
        print(f"Conf shape: {conf.shape}")
        print(f"Conf empty: {conf.empty}")
    
    # Only append if conf is a non-empty DataFrame
    if isinstance(conf, pd.DataFrame) and not conf.empty:
        confs.append(conf)
        print(f"Added confidence data with shape: {conf.shape}")
    else:
        print("Skipping empty or invalid confidence data")

# Debug output before concatenation
print(f"Number of confidence DataFrames collected: {len(confs)}")
for i, df in enumerate(confs):
    print(f"DataFrame {i} shape: {df.shape}")

# Only try to concatenate if we have DataFrames
if confs:
    conf = pd.concat(confs, axis=0)
    print("Final confidence shape:", conf.shape)
else:
    print("Warning: No confidence data to concatenate")
    conf = pd.DataFrame(columns=["match", "correct_prediction", "confidence", "PSW"])

## We add the date to the confidence dataset (can be useful for analysis later)
dates=data.Date.reset_index()
dates.columns=["match","date"]
conf=conf.merge(dates,on="match")

names=data.Winner.reset_index()
names.columns=["match","winner"]
conf=conf.merge(names,on="match")

conf=conf.sort_values("confidence",ascending=False)
conf=conf.reset_index(drop=True)

## We store this dataset
conf.to_csv(CONFIDENCE_DATA_PATH,index=False)
stamp = datetime.datetime.now()
stamp_string = stamp.strftime('%M-%D')
################################################################ END RUN 1 ############################