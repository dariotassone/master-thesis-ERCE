# -*- coding: utf-8 -*-
"""
@author: Dario Tassone
"""
import numpy as np
import pandas as pd
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from functions import *
from functions import robust_eigenvalue_modifier
from functions import resampled_eigenvalue_modifier
from functions import LedoitWolf
from functions import LW_nonlin_shrink
from functions import oos_performance_analysis
# Figure 8: Out-of-Sample Allocation Distribution for MVPs and IPPs
# Load data
return_data = pd.read_csv('return_data.csv')
return_data.set_index('Date', inplace=True, drop=True)
rf_df = return_data.iloc[:, -1].to_frame()
return_data = return_data.iloc[:,:-1]

# Specify parameters and data subset
nMet = 5
estimation_window = 120
holding_window = 12
k = 50
return_data_subset = return_data.sample(n=k, axis=1, random_state=10)

# Compute results for MVP
output_MVP = oos_performance_analysis(return_data = return_data_subset, rf_df = rf_df, estimation_window = 120, holding_window=12, nMet=5, short_constraint=False, MVP=True)
weight_MVP = output_MVP['w']
# Compute results for IPP
output = oos_performance_analysis(return_data = return_data_subset, rf_df = rf_df, estimation_window = 120, holding_window=12, nMet=5, short_constraint=False, MVP=False)
weight = output['w']


method = ['sample', 'ERCE', 'RERCE', 'LW_lin', 'LW_nonlin', 'EW']

# Iteratively plot boxplots of weights over time for different methods and portfolios
for w_i,w in enumerate([weight_MVP,weight]):
    
    # Determine max. and min. weight in three-dim. array
    y_min = w.min()
    y_max = w.max()
    
    # Specify x-axis labels
    years = [2004 + i for i in range(w.shape[1])]
    
    # Generate boxplots for weight matrix of each methodology
    for i in range(5):
        # Extract weight matrix for i-th method
        matrix = w[:, :, i]
        
        # Create a figure for the current matrix
        plt.figure(figsize=(12, 6))
        # Specify font sizes manually
        plt.rcParams.update({
            'axes.labelsize': 18,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18
        })

        # Horizontal line highlighting equal weight level
        plt.axhline(y=1/matrix.shape[0], color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        
        plt.legend()
    
        # Specify marker characteristics
        boxplot = plt.boxplot(matrix, patch_artist=True, flierprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markersize": 5
        })
        
        # Customize box colors (set to grey)
        for box in boxplot['boxes']:
            box.set(facecolor='lightgrey')  
            
        # Customize median 
        for median in boxplot['medians']:
            median.set(color='black', linewidth=2)
        
        # Remove whiskers
        for cap in boxplot['caps']:
            cap.set(visible=False)
                    
        # Show only every 3rd x-axis label
        plt.xticks(
            ticks=range(1, 1+w.shape[1], 3), 
            labels=[years[j] for j in range(0, len(years), 3)], fontsize=18
        )
        
        # Specify y-axis label
        plt.ylim(y_min-0.05, y_max+0.05)
    
        # Add a grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
        # Save as figure
        #filename = f"weights_{w_i}_method_{method[i]}_boxplot.png"
        #plt.savefig(filename, bbox_inches='tight')
        plt.close()
