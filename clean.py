#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import sys
from config import DEFAULT_RANKING

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('tennis_data_cleaner')

def prepare_tennis_data(data_list):
    """
    Prepare and clean tennis match data from multiple sources.
    
    Args:
        data_list: List of DataFrames containing tennis match data
        
    Returns:
        data: Cleaned and concatenated DataFrame or None if data_list is empty
    """
    # Log input data information
    logger.debug(f"prepare_tennis_data received data_list of type: {type(data_list)}")
    logger.debug(f"data_list length: {len(data_list) if isinstance(data_list, list) else 'Not a list'}")
    
    # Check if data_list is empty
    if not data_list:
        logger.warning("Empty data_list provided. Returning None.")
        return None
    
    # Define the columns we want to keep
    wanted_columns = [
        'Date', 'Surface', 'Round', 'Best of', 'Winner', 'Loser', 'WRank', 'LRank', 'WPts', 'LPts', 'Wsets', 'Lsets', 'Comment', 'PSW', 'PSL'
    ]
    
    # Log columns for each DataFrame
    for i, df in enumerate(data_list):
        logger.debug(f"DataFrame {i} columns: {list(df.columns)}")
        print(df.head())
        print(df.tail())
    
    # Process each DataFrame to include only wanted columns
    processed_data_list = []
    for i, df in enumerate(data_list):
        # First, ensure PSW and PSL columns exist (add if missing)
        if 'PSW' not in df.columns:
            df['PSW'] = np.nan
        if 'PSL' not in df.columns:
            df['PSL'] = np.nan
        
        # Select only columns that exist in both the wanted list and the DataFrame
        existing_wanted_columns = [col for col in wanted_columns if col in df.columns]
        
        # Log which columns are missing from this DataFrame
        missing_columns = [col for col in wanted_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"DataFrame {i} is missing these wanted columns: {missing_columns}")
        
        # Create a new DataFrame with only the wanted columns that exist
        new_df = df[existing_wanted_columns].copy()
        
        # Add any missing wanted columns as NaN
        for col in wanted_columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
                
        # Ensure columns are in the desired order
        new_df = new_df[wanted_columns]
        
        processed_data_list.append(new_df)
        logger.debug(f"Processed DataFrame {i} now has columns: {list(new_df.columns)}")
    
    # Concatenate all data
    logger.debug("Concatenating data...")
    data = pd.concat(processed_data_list, axis=0)
    
    # Clean the data
    logger.debug("Cleaning data...")
    data = clean_tennis_ranks_and_sets(data)
    
    logger.debug(f"Final data shape: {data.shape}")
    return data

def clean_tennis_ranks_and_sets(data):
    """
    Clean tennis match data.
    
    Args:
        data: DataFrame containing tennis match data
        
    Returns:
        data: Cleaned DataFrame
    """


    # Sort by date
    data = data.sort_values("Date")
    
    # Clean player rankings
    data["WRank"] = data["WRank"].replace(np.nan, DEFAULT_RANKING)
    data["WRank"] = data["WRank"].replace("NR", DEFAULT_RANKING)

    data["LRank"] = data["LRank"].replace(np.nan, DEFAULT_RANKING)
    data["LRank"] = data["LRank"].replace("NR", DEFAULT_RANKING)

    data["Wsets"] = data["Wsets"].replace(np.nan, 1)
    data["Lsets"] = data["Lsets"].replace(np.nan, 0)
    # Clean set scores

    # Handle player points - first check if these columns exist
    if "WPts" in data.columns:
        # Fill NaN values with 0
        data["WPts"] = data["WPts"].fillna(0)
        # Round to nearest integer before converting
        data["WPts"] = data["WPts"].round().astype(int)
    
    if "LPts" in data.columns:
        # Fill NaN values with 0
        data["LPts"] = data["LPts"].fillna(0)
        # Round to nearest integer before converting
        data["LPts"] = data["LPts"].round().astype(int)

    
    # Reset index
    data = data.reset_index(drop=True)
    data["WRank"] = data["WRank"].astype(int)
    data["LRank"] = data["LRank"].astype(int)
    data["Wsets"] = data["Wsets"].astype(int)
    data["Lsets"] = data["Lsets"].astype(int)

    data["WPts"] = data["WPts"].astype(int)
    data["LPts"] = data["LPts"].astype(int)

    logger.info("Data types after cleaning:")
    for col in data.columns:
        # Get unique types in the column
        unique_types = set(type(x).__name__ for x in data[col].dropna().values)
        # Get sample values of each type
        type_samples = {}
        for dtype in unique_types:
            # Find first value of this type
            for val in data[col].dropna().values:
                if type(val).__name__ == dtype:
                    type_samples[dtype] = val
                    break
        
        logger.info(f"Column '{col}': types={unique_types}, samples={type_samples}")
    
    # Also log pandas dtypes
    logger.info("Pandas dtypes:")
    for col, dtype in data.dtypes.items():
        logger.info(f"Column '{col}': dtype={dtype}")

    
    return data

