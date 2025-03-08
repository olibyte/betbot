#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def profitsAlongTime(conf, matches_delta, bet_percentage=0.35):
    """
    Calculate profits along time by dividing matches into sections.
    
    Args:
        conf: DataFrame containing match data
        matches_delta: Number of matches per section
        bet_percentage: Percentage of matches to bet on (default: 0.35)
        
    Returns:
        profits: Array of profits for each section
        lens: Array of number of matches bet on in each section
    """
    # We bet only on specified percentage of the matches
    confconf = conf.iloc[:int(bet_percentage*len(conf)),:].copy()
    
    span_matches = confconf.match.max() - confconf.match.min() - 1
    N = int(span_matches/matches_delta) + 1
    milestones = np.array([conf.match.min() + matches_delta*i for i in range(N)])
    profits = []
    lens = []
    
    for i in range(N-1):
        beg = milestones[i]
        end = milestones[i+1] - 1
        
        # Fix the boolean indexing by ensuring indices align
        # Method 1: Use .loc for explicit indexing
        mask = (confconf['match'] >= beg) & (confconf['match'] <= end)
        conf_sel = confconf.loc[mask]
        
        # Alternative Method 2: Reset index before filtering
        # confconf_reset = confconf.reset_index(drop=True)
        # conf_sel = confconf_reset[(confconf_reset['match'] >= beg) & (confconf_reset['match'] <= end)]
        
        l = len(conf_sel)
        lens.append(l)
        if l == 0:
            profits.append(0)
        else:    
            p = profitComputation(100, conf_sel)
            profits.append(p)
    
    profits = np.array(profits)
    return profits, lens

def plot_profits_over_time(profits, title="Betting on sections of 100 matches"):
    """
    Plot profits over time.
    
    Args:
        profits: Array of profits
        title: Plot title
    
    Returns:
        fig: The matplotlib figure object
    """
    fig = plt.figure(figsize=(5.5, 3))
    ax = fig.add_axes([0, 0, 1, 0.9])  
    ax.plot(profits, linewidth=2, marker="o")
    plt.suptitle(title)
    ax.set_xlabel("From 2015 to 2022")
    ax.set_ylabel("ROI")
    return fig

def plot_matches_count(lens, title="Betting on sections of 100 matches"):
    """
    Plot number of matches bet on in each section.
    
    Args:
        lens: Array of match counts
        title: Plot title
    
    Returns:
        fig: The matplotlib figure object
    """
    fig = plt.figure(figsize=(5.5, 3))
    ax = fig.add_axes([0, 0, 1, 0.9])  
    ax.plot(lens, linewidth=2, marker="o")
    plt.suptitle(title)
    ax.set_xlabel("From 2015 to 2022")
    ax.set_ylabel("For each section, number of matches we bet on")
    return fig

############################### PROFITS COMPUTING AND VISUALIZATION ############

def profitComputation(percentage,confidence,model_name="0"):
    """
    Input : percentage of matches we want to bet on,confidence dataset
    Output : ROI
    """
    tot_number_matches=len(confidence)
    number_matches_we_bet_on=int(tot_number_matches*(percentage/100))
    matches_selection=confidence.head(number_matches_we_bet_on)
    profit=100*(matches_selection.PSW[matches_selection["correct_prediction"+model_name]==1].sum()-number_matches_we_bet_on)/number_matches_we_bet_on
    return profit

def plotProfits(confidence,title=""):
    """
    Given a confidence dataset, plots the ROI according to the percentage of matches
    we bet on. 
    """
    profits=[]
    ticks=range(5,101)
    for i in ticks:
        p=profitComputation(i,confidence)
        profits.append(p)
    plt.plot(ticks,profits)
    plt.xticks(range(0,101,5))
    plt.xlabel("% of matches we bet on")
    plt.ylabel("Return on investment (%)")
    plt.suptitle(title)

