# -*- coding: utf-8 -*-
"""
@author: Dario Tassone
"""
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions
from scipy.stats import pearsonr
from functions import plot_n_simulation
from functions import covariance_matrix_toeplitz
from functions import multivariate_t
from functions import simulation_result_generator
from sklearn.covariance import LedoitWolf
import random

# Define parameter grids
k_values = [10,50,200]
n_values = [60,120]
nSim = 100
nMet = 5

# Table 1: Monte Carlo Experiment: Toeplitz Population Covariance Matrix with alpha=0.5
results_df_n_05 = simulation_result_generator(
    cov_function=covariance_matrix_toeplitz,
    k_values=k_values,
    n_values=n_values,
    nSim=nSim,
    nMet=nMet,
    dist_function = np.random.multivariate_normal, 
    alpha=0.5,
    rng=454
    )

# Table 2: Monte Carlo Experiment: Toeplitz Population Covariance Matrix with alpha=0.1
results_df_n_01 = simulation_result_generator(
    cov_function=covariance_matrix_toeplitz,
    k_values=k_values,
    n_values=n_values,
    nSim=nSim,
    nMet=nMet,
    dist_function = np.random.multivariate_normal, 
    alpha=0.1,
    rng=454
    )

# Table 9: Monte Carlo Experiment: Toeplitz Population Covariance Matrix with alpha=0.9
results_df_n_09 = simulation_result_generator(
    cov_function=covariance_matrix_toeplitz,
    k_values=k_values,
    n_values=n_values,
    nSim=nSim,
    nMet=nMet,
    dist_function = np.random.multivariate_normal, 
    alpha=0.9,
    rng=454
    )


# Export the DataFrames to Latex
with open('sba_01.tex', 'w') as f:
    f.write(results_df_n_01.to_latex(index=False, float_format="%.4f"))

with open('sba_05.tex', 'w') as f:
    f.write(results_df_n_05.to_latex(index=False, float_format="%.4f"))

with open('sba_09.tex', 'w') as f:
    f.write(results_df_n_09.to_latex(index=False, float_format="%.4f"))


# Appendix: Figure 6: Influence of Simulation Iterations on Performance Metrics
# Specify parameters
nSim_values = [5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300]
k_values = [10]
n_values = [100]
nMet = 5

# Pre-allocate result objects
MPVD = {i: [] for i in range(5)}
WD = {i: [] for i in range(5)}


# Iterate through nSim_values
for nSim in nSim_values:
    # Compute result for each number of simulation (under the specific parameters)
    results_df = simulation_result_generator(
        cov_function=covariance_matrix_toeplitz,
        k_values=k_values,
        n_values=n_values,
        nSim=nSim,
        nMet=nMet,
        dist_function=np.random.multivariate_normal,
        alpha=0.5,
        rng=454
    )
    # Collect value of each method
    for i in range(nMet):
        WD[i].append(results_df["WD"].iloc[i])
        MPVD[i].append(results_df["MVPD"].iloc[i])


# Plot the metrics
plt.rcParams.update({'font.size': 16})
plot_n_simulation(nSim_values, WD, 'WD', legend_flag=False)
plot_n_simulation(nSim_values, MPVD, 'MVPD')

