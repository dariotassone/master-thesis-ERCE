# -*- coding: utf-8 -*-
"""
@author: Dario Tassone
"""
import numpy as np
import pandas as pd
import os
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from functions import *
from functions import robust_eigenvalue_modifier
from functions import resampled_eigenvalue_modifier
from functions import LedoitWolf
from functions import LW_nonlin_shrink
from functions import oos_performance_analysis
# Load data
return_data = pd.read_csv('return_data.csv')
return_data.set_index('Date', inplace=True, drop=True)

rf_df = return_data.iloc[:, -1].to_frame()
return_data = return_data.iloc[:,:-1]

# Specify parameters
nMet = 5
estimation_window = [60,120]
holding_window = [1,3]
k_values = [50,100,200]

# Generation of Tables 4-8 and Tables 10-13
# Iterate through different holding window lenghts
for h in holding_window:
    # Pre-allocate lists
    structure_MVP = []
    performance_MVP = []
    weights_MVP = []
    portfolio_gross_return_cum_tc_MVP = []


    structure_IPP = []
    performance_IPP = []
    weights_IPP = []
    portfolio_gross_return_cum_tc_IPP = []
    
    # Iterate through different estimation window lenghts
    for e in estimation_window:
        # Iterate through different number of assets (investment universes)
        for k in k_values:
            
            # Select k assets
            return_data_subset = return_data.sample(n=k, axis=1, random_state=10)
            
            # Compute results for MVP
            output_MVP = oos_performance_analysis(return_data_subset, rf_df, e, h, nMet, short_constraint=False, MVP=True)
        
            structure_MVP.append(output_MVP['df_structure'])
            performance_MVP.append(output_MVP['df_performance'])
            weights_MVP.append(output_MVP['w'])
            portfolio_gross_return_cum_tc_MVP.append(output_MVP['monthly_return_cum'])
            
            # Compute results matching in-sample with EW
            output_IPP = oos_performance_analysis(return_data_subset, rf_df, e, h, nMet, short_constraint=False, MVP=False)
            
            structure_IPP.append(output_IPP['df_structure'])
            performance_IPP.append(output_IPP['df_performance'])
            weights_IPP.append(output_IPP['w'])
            portfolio_gross_return_cum_tc_IPP.append(output_IPP['monthly_return_cum_tc'])

    globals()[f"df_structure_MVP_{h}"] = pd.concat(structure_MVP, axis=0)
    globals()[f"df_performance_MVP_{h}"] = pd.concat(performance_MVP, axis=0)

    globals()[f"df_structure_IPP_{h}"] = pd.concat(structure_IPP, axis=0)
    globals()[f"df_performance_IPP_{h}"] = pd.concat(performance_IPP, axis=0)

            

df_list_1 = [df_structure_MVP_1, df_performance_MVP_1, df_structure_IPP_1, df_performance_IPP_1]
df_list_3 = [df_structure_MVP_3, df_performance_MVP_3, df_structure_IPP_3, df_performance_IPP_3]

def df_to_latex(df_list):
    """
    Process a list of DataFrames and export them to LaTeX files using their original variable names.
    
    Parameters:
    df_list (list): List of DataFrames to process.
    
    Returns:
    list: List of processed DataFrames.
    """
    processed_dfs = []

    # Iterate through the DataFrames
    for df in df_list:
        # Get the name of the DataFrame variable dynamically
        df_name = [name for name, obj in globals().items() if obj is df]
        filename = os.path.join('OOS_Results', f"{df_name[0]}.tex")  # Use the variable name for the file


        # Process the DataFrame
        df_processed = df.copy()
        df_processed.iloc[:, 3:] = df_processed.iloc[:, 3:].round(4)
        df_processed.iloc[:, 3:] = df_processed.iloc[:, 3:].applymap(lambda x: f"{x:.4f}")
        df_processed = df_processed.replace({"0.0000": "0", "-0.0000": "0"})
        processed_dfs.append(df_processed)

        df_processed.to_latex(filename, index=False)    
    return processed_dfs 

df_to_latex(df_list_1)
df_to_latex(df_list_3)
