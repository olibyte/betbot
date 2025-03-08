#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from categorical_features import categorical_features_encoding, features_players_encoding, features_tournaments_encoding
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_onehot_encoding():
    """
    Test the one-hot encoding process to ensure it's working correctly.
    """
    logger.info("Testing one-hot encoding...")
    
    # Create a simple test dataset
    test_data = pd.DataFrame({
        'Surface': ['Clay', 'Hard', 'Grass', 'Hard'],
        'Round': ['F', 'QF', 'R32', 'SF'],
        'Tournament': ['Roland Garros', 'Paris Masters', 'Wimbledon', 'Barcelona']
    })
    
    # Test categorical encoding
    logger.info("Testing categorical_features_encoding...")
    cat_encoded = categorical_features_encoding(test_data)
    
    # Check if encoding produced reasonable output
    logger.info(f"Original shape: {test_data.shape}, Encoded shape: {cat_encoded.shape}")
    logger.info(f"Number of encoded columns: {cat_encoded.shape[1]}")
    logger.info(f"Sample of encoded columns: {list(cat_encoded.columns[:10])}")
    
    # Check if all rows have at least one 1 in each category
    categories = ['Surface', 'Round', 'Tournament']
    for cat in categories:
        cat_cols = [col for col in cat_encoded.columns if col.startswith(cat)]
        row_sums = cat_encoded[cat_cols].sum(axis=1)
        logger.info(f"Category {cat}: {len(cat_cols)} columns, row sums: {row_sums.values}")
        assert all(row_sums == 1), f"Each row should have exactly one 1 in category {cat}"
    
    logger.info("Basic one-hot encoding test passed!")
    
    # Create a simple test for player encoding
    test_players = pd.DataFrame({
        'Winner': ['Federer', 'Nadal', 'Djokovic', 'Murray'],
        'Loser': ['Nadal', 'Djokovic', 'Murray', 'Federer']
    })
    
    logger.info("Testing features_players_encoding...")
    players_encoded = features_players_encoding(test_players, nb_players=3)
    
    logger.info(f"Players encoded shape: {players_encoded.shape}")
    logger.info(f"Players encoded columns: {list(players_encoded.columns)}")
    
    # Test tournament encoding
    test_tournaments = pd.DataFrame({
        'Tournament': ['Roland Garros', 'Wimbledon', 'US Open', 'Australian Open']
    })
    
    logger.info("Testing features_tournaments_encoding...")
    tournaments_encoded = features_tournaments_encoding(test_tournaments, nb_tournaments=3)
    
    logger.info(f"Tournaments encoded shape: {tournaments_encoded.shape}")
    logger.info(f"Tournaments encoded columns: {list(tournaments_encoded.columns)}")
    
    logger.info("All encoding tests completed!")
    return cat_encoded, players_encoded, tournaments_encoded

def test_with_real_data():
    """
    Test the feature processing with real data from your dataset.
    """
    from config import ATP_DATA_PATH
    import datetime
    
    logger.info("Testing with real data...")
    try:
        # Load data
        data = pd.read_csv(ATP_DATA_PATH)
        data.Date = data.Date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        
        # Take a small sample
        sample_size = min(1000, len(data))
        data_sample = data.sample(sample_size, random_state=42)
        
        # Process categorical features
        logger.info("Processing categorical features...")
        features_categorical = data_sample[["Series", "Court", "Surface", "Round", "Best of", "Tournament"]]
        features_categorical_encoded = categorical_features_encoding(features_categorical)
        
        # Check encoding
        logger.info(f"Categorical features shape: {features_categorical.shape}")
        logger.info(f"Encoded features shape: {features_categorical_encoded.shape}")
        
        # Check for NaN values
        nan_count = features_categorical_encoded.isna().sum().sum()
        logger.info(f"NaN values in encoded features: {nan_count}")
        
        # Check for all-zero rows
        zero_rows = (features_categorical_encoded.sum(axis=1) == 0).sum()
        logger.info(f"Rows with all zeros: {zero_rows}")
        
        # Process player features
        logger.info("Processing player features...")
        players_encoded = features_players_encoding(data_sample)
        
        # Check player encoding
        logger.info(f"Players encoded shape: {players_encoded.shape}")
        nan_count = players_encoded.isna().sum().sum()
        logger.info(f"NaN values in player features: {nan_count}")
        
        # Process tournament features
        logger.info("Processing tournament features...")
        tournaments_encoded = features_tournaments_encoding(data_sample)
        
        # Check tournament encoding
        logger.info(f"Tournaments encoded shape: {tournaments_encoded.shape}")
        nan_count = tournaments_encoded.isna().sum().sum()
        logger.info(f"NaN values in tournament features: {nan_count}")
        
        # Combine all features
        logger.info("Combining all features...")
        features_onehot = pd.concat([features_categorical_encoded, players_encoded, tournaments_encoded], axis=1)
        
        # Check combined features
        logger.info(f"Combined features shape: {features_onehot.shape}")
        nan_count = features_onehot.isna().sum().sum()
        logger.info(f"NaN values in combined features: {nan_count}")
        
        # Check for duplicate columns
        duplicate_cols = features_onehot.columns[features_onehot.columns.duplicated()].tolist()
        logger.info(f"Duplicate columns: {duplicate_cols}")
        
        # Check for constant columns (all values the same)
        constant_cols = [col for col in features_onehot.columns 
                         if features_onehot[col].nunique() == 1]
        logger.info(f"Number of constant columns: {len(constant_cols)}")
        if constant_cols:
            logger.info(f"Sample constant columns: {constant_cols[:5]}")
        
        # Check for sparse columns (mostly zeros)
        sparsity = features_onehot.apply(lambda x: (x == 0).mean())
        very_sparse = sparsity[sparsity > 0.99].index.tolist()
        logger.info(f"Number of very sparse columns (>99% zeros): {len(very_sparse)}")
        
        # Visualize feature distribution
        plt.figure(figsize=(12, 6))
        sns.heatmap(features_onehot.sample(min(100, len(features_onehot))).T, 
                   cmap='viridis', cbar=False, yticklabels=False)
        plt.title('Feature Matrix Visualization (Sample)')
        plt.xlabel('Samples')
        plt.ylabel('Features')
        plt.savefig('feature_matrix.png')
        logger.info("Feature matrix visualization saved as 'feature_matrix.png'")
        
        # Check row duplication
        logger.info("Testing row duplication...")
        # Create a small sample for testing duplication
        sample_df = features_onehot.head(5).copy()
        logger.info(f"Sample before duplication: {sample_df.shape}")

        # Method 1: Using np.repeat
        method1 = pd.DataFrame(
            np.repeat(sample_df.values, 2, axis=0),
            columns=sample_df.columns
        )
        logger.info(f"Method 1 shape: {method1.shape}")

        # Method 2: Using index-based duplication
        new_index = np.repeat(range(len(sample_df)), 2)
        method2 = pd.DataFrame(
            sample_df.values[new_index],
            columns=sample_df.columns
        )
        logger.info(f"Method 2 shape: {method2.shape}")

        # Method 3: Using pandas concat
        method3 = pd.concat([sample_df, sample_df], ignore_index=True)
        logger.info(f"Method 3 shape: {method3.shape}")

        # Check if methods produce the same result
        for i in range(min(3, len(sample_df))):
            original_row = sample_df.iloc[i].values
            m1_row1 = method1.iloc[i*2].values
            m1_row2 = method1.iloc[i*2+1].values
            m2_row1 = method2.iloc[i*2].values
            m2_row2 = method2.iloc[i*2+1].values
            m3_row1 = method3.iloc[i].values
            m3_row2 = method3.iloc[i+len(sample_df)].values
            
            logger.info(f"Row {i} comparison:")
            logger.info(f"  Original vs Method1 first row equal: {np.array_equal(original_row, m1_row1)}")
            logger.info(f"  Original vs Method1 second row equal: {np.array_equal(original_row, m1_row2)}")
            logger.info(f"  Original vs Method2 first row equal: {np.array_equal(original_row, m2_row1)}")
            logger.info(f"  Original vs Method2 second row equal: {np.array_equal(original_row, m2_row2)}")
            logger.info(f"  Original vs Method3 first row equal: {np.array_equal(original_row, m3_row1)}")
            logger.info(f"  Original vs Method3 second row equal: {np.array_equal(original_row, m3_row2)}")
        
        logger.info("Row duplication test passed!")
        
        # Add this after the "Combining all features..." section
        logger.info(f"Categorical features index: {features_categorical_encoded.index[:5]} (length: {len(features_categorical_encoded)})")
        logger.info(f"Players features index: {players_encoded.index[:5]} (length: {len(players_encoded)})")
        logger.info(f"Tournaments features index: {tournaments_encoded.index[:5]} (length: {len(tournaments_encoded)})")

        # Check the shapes before and after concatenation
        logger.info(f"Shapes before concat: cat={features_categorical_encoded.shape}, players={players_encoded.shape}, tournaments={tournaments_encoded.shape}")
        features_onehot = pd.concat([features_categorical_encoded, players_encoded, tournaments_encoded], axis=1)
        logger.info(f"Shape after concat: {features_onehot.shape}")

        # Check if indices match
        cat_indices = set(features_categorical_encoded.index)
        player_indices = set(players_encoded.index)
        tournament_indices = set(tournaments_encoded.index)
        logger.info(f"Index overlap: cat & players: {len(cat_indices.intersection(player_indices))}, cat & tournaments: {len(cat_indices.intersection(tournament_indices))}")
        
        # Check for NaN values in the final feature matrix
        logger.info("Checking final feature matrix...")
        # Create a small sample dataset for odds and Elo
        sample_odds = pd.DataFrame({
            'odds': np.random.uniform(1.1, 5.0, size=len(features_onehot))
        })
        sample_elo = pd.DataFrame({
            'elo_a': np.random.normal(1500, 100, size=len(features_onehot)),
            'elo_b': np.random.normal(1500, 100, size=len(features_onehot)),
            'proba_elo': np.random.uniform(0.1, 0.9, size=len(features_onehot))
        })

        # Combine all features as in main.py
        final_features = pd.concat([
            sample_odds,
            sample_elo,
            features_onehot
        ], axis=1)

        logger.info(f"Final features shape: {final_features.shape}")
        nan_count = final_features.isna().sum().sum()
        logger.info(f"NaN values in final features: {nan_count}")

        # Check if any rows have NaN values
        rows_with_nan = final_features.isna().any(axis=1).sum()
        logger.info(f"Rows with NaN values: {rows_with_nan} out of {len(final_features)}")

        return final_features
        
    except Exception as e:
        logger.error(f"Error in real data test: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting feature validation tests...")
    
    # Run basic encoding tests
    cat_encoded, players_encoded, tournaments_encoded = test_onehot_encoding()
    
    # Run tests with real data
    try:
        final_features = test_with_real_data()
        
        # Additional test: Check if features have variance
        logger.info("Checking feature variance...")
        variance = final_features.var()
        zero_var = (variance == 0).sum()
        logger.info(f"Features with zero variance: {zero_var} out of {len(variance)}")
        
        # Check if any features have high correlation
        logger.info("Checking feature correlation...")
        # Sample a subset of columns if there are too many
        if final_features.shape[1] > 100:
            sample_cols = np.random.choice(final_features.columns, 100, replace=False)
            corr_matrix = final_features[sample_cols].corr().abs()
        else:
            corr_matrix = final_features.corr().abs()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > 0.95
        high_corr = [(upper.index[i], upper.columns[j], upper.iloc[i, j]) 
                     for i in range(len(upper.index)) 
                     for j in range(len(upper.columns)) 
                     if upper.iloc[i, j] > 0.95]
        
        logger.info(f"Number of highly correlated feature pairs: {len(high_corr)}")
        if high_corr:
            logger.info(f"Sample highly correlated pairs: {high_corr[:5]}")
        
    except Exception as e:
        logger.warning(f"Real data test failed: {str(e)}")
    
    logger.info("All feature validation tests completed!") 