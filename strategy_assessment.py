#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import time
from utilities import check_model_params
############################### STRATEGY ASSESSMENT ############################
### the following functions are used to make the predictions and compute the ROI

def xgbModelBinary(xtrain, ytrain, xval, yval, xgb_params, sample_weights=None):
    """
    Train an XGBoost binary classification model.
    
    Args:
        xtrain: Training features
        ytrain: Training labels
        xval: Validation features
        yval: Validation labels
        xgb_params: XGBoost parameters from config
        sample_weights: Optional sample weights
    
    Returns:
        Trained XGBoost model
    """
    # Ensure all features are numeric
    for col in xtrain.columns:
        if xtrain[col].dtype == 'object':
            print(f"Converting column {col} from object to numeric in training data")
            xtrain[col] = pd.to_numeric(xtrain[col], errors='coerce').fillna(0)
            xval[col] = pd.to_numeric(xval[col], errors='coerce').fillna(0)
    
    # Check for any remaining object columns
    object_cols_train = xtrain.select_dtypes(include=['object']).columns.tolist()
    object_cols_val = xval.select_dtypes(include=['object']).columns.tolist()
    
    if object_cols_train or object_cols_val:
        print(f"Warning: Dropping object columns: {object_cols_train}")
        xtrain = xtrain.drop(columns=object_cols_train)
        xval = xval.drop(columns=object_cols_val)
    
    # Create DMatrix
    dtrain = xgb.DMatrix(xtrain, label=ytrain)
    dval = xgb.DMatrix(xval, label=yval)
    
    # Set up evaluation list
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    # Set up parameters
    params = {
        'eval_metric': "logloss",
        'objective': "binary:logistic"
    }
    
    # If xgb_params is a list (from config), extract values
    if isinstance(xgb_params, (list, tuple, np.ndarray)):
        try:
            # Based on the code snippet, parameters are accessed by index
            params.update({
                'eta': xgb_params[0],
                'max_depth': int(xgb_params[1]),
                'min_child_weight': xgb_params[2],
                'gamma': xgb_params[3],
                'colsample_bytree': xgb_params[4],
                'lambda': xgb_params[5],
                'alpha': xgb_params[6]
            })
            num_round = int(xgb_params[7])
            early_stopping_rounds = int(xgb_params[8])
            
            # Add subsample if available
            if len(xgb_params) > 9:
                params['subsample'] = xgb_params[9]
        except (IndexError, TypeError) as e:
            print(f"Error extracting parameters from config: {e}")
            print("Using default parameters")
            num_round = 1000
            early_stopping_rounds = 10
    else:
        # If xgb_params is a dictionary, use it directly
        print("Using parameters from dictionary")
        params.update(xgb_params)
        num_round = params.pop('num_round', 1000)
        early_stopping_rounds = params.pop('early_stopping_rounds', 10)
        
    # Train the model
    check_model_params(params)
    bst = xgb.train(params, dtrain, num_round, evallist, 
                   early_stopping_rounds=early_stopping_rounds, 
                   verbose_eval=False)
    
    return bst


def mer(t):
    # If more than half the models choose the right outcome for the match, we can say
    # in real situation we would have been right. Otherwise wrong.
    # And the confidence in the chosen outcome is the mean of the confidences of the models
    # that chose this outcome.
    w=np.array([t[0],t[1],t[2],t[3],t[4],t[5],t[6]]).astype(bool)
    conf=np.array([t[7],t[8],t[9],t[10],t[11],t[12],t[13]])
    if w.sum()>=4:
        return 1,conf[w].mean()
    else:
        return 0,conf[~w].mean() 
    
def simple_strategy(test_beginning_match, duration_train_matches, duration_val_matches, 
                   duration_test_matches, xgb_params, nb_players, nb_tournaments, features, data):
    """Simplified strategy assessment function"""
    try:
        # Number of matches in dataset
        nm = len(data)
        print(f"Total matches in data: {nm}")
        
        # Calculate indices with bounds checking
        beg_test = test_beginning_match
        end_test = min(beg_test + duration_test_matches - 1, nm - 1)
        
        # Ensure we have enough data for validation
        beg_val = beg_test - duration_val_matches
        if beg_val < 0:
            print(f"Warning: Not enough data for validation. Need {duration_val_matches} matches before {beg_test}")
            return pd.DataFrame(columns=["match", "correct_prediction", "confidence", "PSW"])
        end_val = beg_test - 1
        
        # Ensure we have enough data for training
        beg_train = beg_val - duration_train_matches
        if beg_train < 0:
            print(f"Warning: Not enough data for training. Need {duration_train_matches} matches before {beg_val}")
            return pd.DataFrame(columns=["match", "correct_prediction", "confidence", "PSW"])
        end_train = beg_val - 1
        
        print(f"Train: {beg_train}-{end_train}, Val: {beg_val}-{end_val}, Test: {beg_test}-{end_test}")
        
        # Create train/val/test sets
        X_train = features.iloc[beg_train:end_train+1]
        y_train = pd.Series([1, 0] * ((end_train - beg_train + 1) // 2))
        
        X_val = features.iloc[beg_val:end_val+1]
        y_val = pd.Series([1, 0] * ((end_val - beg_val + 1) // 2))
        
        X_test = features.iloc[beg_test:end_test+1]
        y_test = pd.Series([1, 0] * ((end_test - beg_test + 1) // 2))
        
        # Train model
        model = xgbModelBinary(X_train, y_train, X_val, y_val, xgb_params)
        
        # Make predictions
        preds = make_predictions(model, X_test)
        
        # Process predictions - FIXED VERSION
        matches = range(beg_test, end_test + 1)
        binary_preds = [1 if p > 0.5 else 0 for p in preds]
        
        # Check if predictions are correct by comparing to actual outcomes
        correct_predictions = [1 if pred == actual else 0 
                              for pred, actual in zip(binary_preds, y_test)]
        
        # Create confidence DataFrame
        conf_df = pd.DataFrame({
            "match": matches,
            "correct_prediction": correct_predictions,  # Now this is actual correctness
            "confidence": preds,
            "PSW": data.iloc[beg_test:end_test+1]["PSW"].values
        })
        
        print(f"Created confidence DataFrame with shape: {conf_df.shape}")
        
        # Check for duplicated match IDs
        train_matches = set(range(beg_train, end_train+1))
        test_matches = set(range(beg_test, end_test+1))
        overlap = train_matches.intersection(test_matches)
        if overlap:
            print(f"ERROR: {len(overlap)} matches appear in both training and test sets!")
        
        # Add verification
        verify_symmetric_labels(X_train, y_train)
        verify_symmetric_labels(X_val, y_val)
        
        return conf_df
        
    except Exception as e:
        print(f"Error in simple_strategy: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["match", "correct_prediction", "confidence", "PSW"])

def verify_symmetric_labels(features, labels):
    """
    Verify that labels are correctly assigned for symmetric match representations.
    
    Args:
        features: DataFrame containing feature data with symmetric representation
        labels: Series containing binary labels (1 for win, 0 for loss)
    
    Returns:
        True if labels pass verification, False otherwise
    """
    print("Verifying symmetric labels...")
    
    # Check that we have an even number of rows (each match appears twice)
    if len(features) % 2 != 0:
        print(f"ERROR: Odd number of rows ({len(features)}). Each match should appear twice.")
        return False
    
    errors = 0
    checked_pairs = 0
    
    # For each pair of rows (original and swapped)
    for i in range(0, len(features), 2):
        if i+1 >= len(features):
            break
            
        checked_pairs += 1
        
        # 1. Check that labels sum to 1 (one win, one loss)
        label_sum = labels.iloc[i] + labels.iloc[i+1]
        if label_sum != 1:
            print(f"ERROR: Match pair at indices {i},{i+1} has incorrect label sum: {label_sum}")
            errors += 1
            continue
        
        # 2. Check that Elo ratings are swapped
        if 'elo_a' in features.columns and 'elo_b' in features.columns:
            if features.iloc[i]['elo_a'] != features.iloc[i+1]['elo_b'] or \
               features.iloc[i]['elo_b'] != features.iloc[i+1]['elo_a']:
                print(f"ERROR: Elo ratings not properly swapped for match pair at indices {i},{i+1}")
                print(f"  Row {i}: elo_a={features.iloc[i]['elo_a']}, elo_b={features.iloc[i]['elo_b']}")
                print(f"  Row {i+1}: elo_a={features.iloc[i+1]['elo_a']}, elo_b={features.iloc[i+1]['elo_b']}")
                errors += 1
                continue
        
        # 3. Check that probabilities sum to approximately 1
        if 'proba_elo' in features.columns:
            prob_sum = features.iloc[i]['proba_elo'] + features.iloc[i+1]['proba_elo']
            if not (0.99 <= prob_sum <= 1.01):
                print(f"ERROR: Probabilities don't sum to 1 for match pair at indices {i},{i+1}: {prob_sum}")
                errors += 1
                continue
    
    # Report results
    if errors == 0:
        print(f"✓ All {checked_pairs} match pairs passed verification!")
        return True
    else:
        print(f"✗ Found {errors} errors in {checked_pairs} match pairs.")
        return False

def make_predictions(model, xtest):
    # This should be how predictions are made
    dtest = xgb.DMatrix(xtest)
    preds = model.predict(dtest)
    
    # Check prediction distribution
    print(f"Prediction mean: {preds.mean():.4f}")
    print(f"Prediction std: {preds.std():.4f}")
    print(f"Predictions > 0.9: {(preds > 0.9).mean():.4f}")
    
    # A healthy model should have diverse predictions
    if preds.std() < 0.1:
        print("WARNING: Very low prediction variance, model might not be learning")
        return False
    
    return preds
