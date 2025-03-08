#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from config import CONFIDENCE_DATA_PATH
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tennis_analysis')

def calculate_roi(confidence_data_path, stake=1.0, confidence_threshold=None):
    """
    Calculate Return on Investment (ROI) from betting predictions.
    
    Args:
        confidence_data_path (str): Path to the confidence data CSV file
        stake (float): Amount staked on each bet (default: 1.0)
        confidence_threshold (float, optional): Only consider bets with confidence above this threshold
        
    Returns:
        dict: Dictionary containing ROI metrics and statistics
    """
    # Load confidence data
    logger.info(f"Loading confidence data from {confidence_data_path}")
    try:
        data = pd.read_csv(confidence_data_path)
        logger.info(f"Loaded {len(data)} predictions")
    except Exception as e:
        logger.error(f"Error loading confidence data: {str(e)}")
        return None
    
    # Apply confidence threshold if specified
    if confidence_threshold is not None:
        original_count = len(data)
        data = data[data['confidence'] >= confidence_threshold]
        logger.info(f"Applied confidence threshold {confidence_threshold}: {len(data)}/{original_count} predictions remain")
    
    # Calculate returns for each bet
    data['return'] = np.where(data['correct_prediction'] == 1, (data['PSW'] - 1) * stake, -stake)
    
    # Calculate overall metrics
    total_bets = len(data)
    winning_bets = data['correct_prediction'].sum()
    losing_bets = total_bets - winning_bets
    
    total_investment = total_bets * stake
    total_return = data['return'].sum()
    
    roi = (total_return / total_investment) * 100 if total_investment > 0 else 0
    
    win_rate = (winning_bets / total_bets) * 100 if total_bets > 0 else 0
    
    # Calculate average odds
    avg_odds = data['PSW'].mean()
    avg_winning_odds = data.loc[data['correct_prediction'] == 1, 'PSW'].mean() if winning_bets > 0 else 0
    
    # Calculate metrics by confidence level
    if 'confidence' in data.columns:
        data['confidence_bin'] = pd.cut(data['confidence'], bins=10)
        roi_by_confidence = data.groupby('confidence_bin').apply(
            lambda x: (x['return'].sum() / (len(x) * stake) * 100) if len(x) > 0 else 0
        ).to_dict()
    else:
        roi_by_confidence = {}
    
    # Prepare results
    results = {
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'losing_bets': losing_bets,
        'win_rate': win_rate,
        'total_investment': total_investment,
        'total_return': total_return,
        'roi': roi,
        'avg_odds': avg_odds,
        'avg_winning_odds': avg_winning_odds,
        'roi_by_confidence': roi_by_confidence
    }
    
    # Log summary
    logger.info(f"ROI Analysis Summary:")
    logger.info(f"Total bets: {total_bets}")
    logger.info(f"Win rate: {win_rate:.2f}%")
    logger.info(f"ROI: {roi:.2f}%")
    logger.info(f"Total profit/loss: {total_return:.2f} units")
    
    return results

def analyze_by_year(confidence_data_path, stake=1.0):
    """
    Analyze ROI broken down by year.
    
    Args:
        confidence_data_path (str): Path to the confidence data CSV file
        stake (float): Amount staked on each bet (default: 1.0)
        
    Returns:
        dict: Dictionary containing ROI metrics by year
    """
    # Load confidence data
    try:
        data = pd.read_csv(confidence_data_path)
        logger.info(f"Loaded {len(data)} predictions for year analysis")
    except Exception as e:
        logger.error(f"Error loading confidence data: {str(e)}")
        return None
    
    # Ensure date column exists and is in datetime format
    if 'date' not in data.columns:
        logger.error("Date column not found in confidence data")
        return None
    
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    
    # Calculate returns for each bet
    data['return'] = np.where(data['correct_prediction'] == 1, (data['PSW'] - 1) * stake, -stake)
    
    # Group by year and calculate metrics
    yearly_results = {}
    for year, group in data.groupby('year'):
        total_bets = len(group)
        winning_bets = group['correct_prediction'].sum()
        total_return = group['return'].sum()
        roi = (total_return / (total_bets * stake)) * 100
        
        yearly_results[int(year)] = {
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'win_rate': (winning_bets / total_bets) * 100,
            'total_return': total_return,
            'roi': roi
        }
    
    # Log yearly summary
    logger.info(f"Yearly ROI Analysis:")
    for year, metrics in sorted(yearly_results.items()):
        logger.info(f"{year}: ROI = {metrics['roi']:.2f}%, Bets = {metrics['total_bets']}, Win Rate = {metrics['win_rate']:.2f}%")
    
    return yearly_results

def analyze_by_confidence_threshold(confidence_data_path, stake=1.0, thresholds=None):
    """
    Analyze how ROI changes with different confidence thresholds.
    
    Args:
        confidence_data_path (str): Path to the confidence data CSV file
        stake (float): Amount staked on each bet (default: 1.0)
        thresholds (list, optional): List of confidence thresholds to analyze
        
    Returns:
        pd.DataFrame: DataFrame with ROI metrics for each threshold
    """
    if thresholds is None:
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Load confidence data
    try:
        data = pd.read_csv(confidence_data_path)
        logger.info(f"Loaded {len(data)} predictions for threshold analysis")
    except Exception as e:
        logger.error(f"Error loading confidence data: {str(e)}")
        return None
    
    # Calculate returns for each bet
    data['return'] = np.where(data['correct_prediction'] == 1, (data['PSW'] - 1) * stake, -stake)
    
    # Analyze each threshold
    results = []
    for threshold in thresholds:
        filtered_data = data[data['confidence'] >= threshold] if threshold > 0 else data
        
        if len(filtered_data) == 0:
            results.append({
                'threshold': threshold,
                'total_bets': 0,
                'win_rate': 0,
                'roi': 0,
                'total_return': 0
            })
            continue
        
        total_bets = len(filtered_data)
        winning_bets = filtered_data['correct_prediction'].sum()
        win_rate = (winning_bets / total_bets) * 100
        total_return = filtered_data['return'].sum()
        roi = (total_return / (total_bets * stake)) * 100
        
        results.append({
            'threshold': threshold,
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'total_return': total_return
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Log threshold analysis
    logger.info(f"Confidence Threshold Analysis:")
    for _, row in results_df.iterrows():
        logger.info(f"Threshold {row['threshold']}: ROI = {row['roi']:.2f}%, Bets = {row['total_bets']}, Win Rate = {row['win_rate']:.2f}%")
    
    return results_df

if __name__ == "__main__":
    # Example usage
    confidence_data_path = CONFIDENCE_DATA_PATH
    
    # Calculate overall ROI
    overall_roi = calculate_roi(confidence_data_path)
    
    # Analyze by year
    yearly_roi = analyze_by_year(confidence_data_path)
    
    # Analyze by confidence threshold
    threshold_analysis = analyze_by_confidence_threshold(confidence_data_path)
