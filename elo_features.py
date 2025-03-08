#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
import pandas as pd
import numpy as np
from collections import defaultdict

def compute_elo_rankings(data):
    """
    Given the list on matches in chronological order, for each match, computes 
    the elo ranking of the 2 players at the beginning of the match
    
    """
    print("Elo rankings computing...")
    # Sort by date first
    data = data.sort_values('Date')
    
    # Initialize ratings
    player_ratings = defaultdict(lambda: 1500)
    
    # Store pre-match ratings
    elo_a = []
    elo_b = []
    proba_elo = []
    
    for idx, match in data.iterrows():
        # Get current ratings before the match
        rating_winner = player_ratings[match.Winner]
        rating_loser = player_ratings[match.Loser]
        
        # Store pre-match ratings
        elo_a.append(rating_winner)
        elo_b.append(rating_loser)
        
        # Calculate win probability
        proba = 1.0 / (1.0 + 10.0 ** ((rating_loser - rating_winner) / 400.0))
        proba_elo.append(proba)
        
        # Update ratings after the match
        update_elo(player_ratings, match.Winner, match.Loser)
    
    return pd.DataFrame({'elo_a': elo_a, 'elo_b': elo_b, 'proba_elo': proba_elo})

def update_elo(player_ratings, winner, loser, k_factor=32):
    """
    Update Elo ratings for two players after a match.
    
    Args:
        player_ratings: Dictionary mapping player names to their current Elo ratings
        winner: Name of the winning player
        loser: Name of the losing player
        k_factor: How much ratings change after each match (default: 32)
    """
    # Get current ratings
    winner_rating = player_ratings[winner]
    loser_rating = player_ratings[loser]
    
    # Calculate expected win probabilities
    winner_expected = 1.0 / (1.0 + 10.0 ** ((loser_rating - winner_rating) / 400.0))
    loser_expected = 1.0 - winner_expected
    
    # Update ratings
    player_ratings[winner] = winner_rating + k_factor * (1.0 - winner_expected)
    player_ratings[loser] = loser_rating + k_factor * (0.0 - loser_expected)
    
    return player_ratings

def compute_elo_ratings_fixed(data):
    # Sort by date first
    data = data.sort_values('Date')
    
    # Initialize ratings
    player_ratings = defaultdict(lambda: 1500)
    
    # Store pre-match ratings for each match
    match_ratings = {}
    
    for idx, match in data.iterrows():
        # Store CURRENT ratings before the match
        match_ratings[idx] = {
            'player1': match.Winner,
            'player2': match.Loser,
            'elo_player1': player_ratings[match.Winner],  # Current rating BEFORE match
            'elo_player2': player_ratings[match.Loser]    # Current rating BEFORE match
        }
        
        # Update ratings AFTER the match for future matches
        update_elo(player_ratings, match.Winner, match.Loser)
    
    return match_ratings