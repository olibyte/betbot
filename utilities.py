#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import sys
import datetime
import numpy as np
import pandas as pd

def dump(obj,name):
	pickle.dump(obj,open(name+'.p',"wb")) 
def load(name):
	obj=pickle.load( open( name+".p", "rb" ) ) 
	return obj

def log_to_file(log_file_path=None, append=True, include_timestamp=True):


    """
    Redirects terminal output (print statements) to a log file.
    
    Args:
        log_file_path (str): Path to the log file. If None, uses 'output_log.txt'.
        append (bool): If True, appends to existing file. If False, overwrites it.
        include_timestamp (bool): If True, adds timestamp to the log file name.
        
    Returns:
        original_stdout: The original stdout object, which can be used to restore normal printing.
        
    Example:
        original_stdout = log_to_file('my_log.txt')
        print("This will go to the log file")
        # To restore normal printing:
        sys.stdout = original_stdout
    """
    if log_file_path is None:
        log_file_path = 'output_log.txt'
    
    if include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = log_file_path.rsplit('.', 1) if '.' in log_file_path else (log_file_path, 'txt')
        log_file_path = f"{name}_{timestamp}.{ext}"
    
    mode = 'a' if append else 'w'
    
    # Store the original stdout for later restoration
    original_stdout = sys.stdout
    
    # Redirect stdout to the log file
    sys.stdout = open(log_file_path, mode)
    
    return original_stdout

# Check if Elo ratings are calculated using only past matches
def check_temporal_integrity(data, features):
    # Sort data chronologically
    data = data.sort_values('Date')
    
    # Select a test match
    test_match_idx = len(data) - 100  # Near the end of your dataset
    test_match_date = data.iloc[test_match_idx]['Date']
    
    # Get players in this match
    player1 = data.iloc[test_match_idx]['Winner']
    player2 = data.iloc[test_match_idx]['Loser']
    
    # Check if any future matches of these players are used in feature calculation
    future_matches = data[(data['Date'] > test_match_date) & 
                          ((data['Winner'] == player1) | (data['Winner'] == player2) |
                           (data['Loser'] == player1) | (data['Loser'] == player2))]
    
    if not future_matches.empty:
        print(f"WARNING: Found {len(future_matches)} future matches that might affect feature calculation")
        return False
    
    return True

# Check for suspicious feature correlations
def check_feature_correlations(features, labels):
    correlations = []
    
    for col in features.columns:
        corr = np.corrcoef(features[col], labels)[0, 1]
        correlations.append((col, abs(corr)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Print top correlations
    print("Top feature correlations with labels:")
    for col, corr in correlations:
        print(f"{col}: {corr:.4f}")
    
    # Check for suspiciously high correlations
    if correlations[0][1] > 0.8:
        print(f"WARNING: Feature {correlations[0][0]} has suspiciously high correlation: {correlations[0][1]:.4f}")
        return False
    
    return True

# Check XGBoost parameters
def check_model_params(xgb_params):
    # Extract parameters
    if isinstance(xgb_params, (list, tuple, np.ndarray)):
        eta = xgb_params[0]
        max_depth = int(xgb_params[1])
        min_child_weight = xgb_params[2]
        
        # Check for overfitting-prone settings
        if max_depth > 10:
            print(f"WARNING: max_depth={max_depth} is very high, prone to overfitting")
            return False
        
        if min_child_weight < 1:
            print(f"WARNING: min_child_weight={min_child_weight} is very low, prone to overfitting")
            return False
    
    return True
def create_features_no_leakage(data, match_ratings):
   # Create features with NO leakage
   features_rows = []
   labels = []
   
   for idx, match in data.iterrows():
       ratings = match_ratings[idx]
       
       # First representation: player1 vs player2 (as stored)
       features_1 = {
           'elo_a': ratings['elo_player1'],
           'elo_b': ratings['elo_player2'],
           'proba_elo': 1.0 / (1.0 + 10.0 ** ((ratings['elo_player2'] - ratings['elo_player1']) / 400.0)),
           # Add other features
       }
       
       # Second representation: player2 vs player1 (swapped)
       features_2 = {
           'elo_a': ratings['elo_player2'],
           'elo_b': ratings['elo_player1'],
           'proba_elo': 1.0 / (1.0 + 10.0 ** ((ratings['elo_player1'] - ratings['elo_player2']) / 400.0)),
           # Add other features (swapped)
       }
       
       # Add both representations
       features_rows.extend([features_1, features_2])
       
       # Labels: 1 if player1 won (true), 0 if player2 won (false)
       # For first representation: player1 is Winner, so label is 1
       # For second representation: player1 is Loser, so label is 0
       labels.extend([1, 0])
       
   return pd.DataFrame(features_rows), np.array(labels)

def ensure_test_after_train(data, test_data):
    # Ensure test data is strictly after training data
    train_cutoff_date = test_data['Date'].min() - pd.Timedelta(days=1)
    train_data = data[data['Date'] <= train_cutoff_date] 
    return train_data, test_data