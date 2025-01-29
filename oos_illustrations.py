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
from functions import plot_composition_map

# Load data
return_data = pd.read_csv('return_data.csv')
return_data.set_index('Date', inplace=True, drop=True)

# Specify tickers of eleven stocks represented in the eleven GICS sectors
tickers = ['CVX', 'DD', 'DIS', 'DUK', 'GE', 'JNJ', 'JPM', 'KO', 'SPG', 'T', 'MSFT']

# Create the subset of tickers for the two-year period starting in 1994
return_data_subset = return_data.iloc[0:24,:]
rf = return_data_subset.iloc[:,-1].mean()
return_data_subset = return_data_subset[tickers]
return_data_subset_array = np.array(return_data_subset)


# Figure 1: Efficient Frontiers of RERCE and Its Individual ERCEs 
# Compute mean vector as annualized arithmetic mean return
mean = np.array(return_data_subset.mean())*12

# Compute sample covariance matrix, in particular return variances
sample = np.cov(return_data_subset, rowvar=False)
variances = np.diag(sample)

# Compute RERCE and individual ERCEs using a seed that returns illustrative results
result = resampled_eigenvalue_modifier(return_data_subset_array,nSim=10,rng=3)
RERCE = result['cov_est']
cov_list = result['cov_est_sim_list']
cov_list.append(RERCE)

# Pre-allocate return and risk lists
returns = []
volatilities = []

# Resets all rcParams to default values
plt.rcdefaults()
plt.figure(figsize=(10, 6))

# Iteratively generate efficient frontiers for RERCE and each ERCE
for i, est in enumerate(cov_list):
    ef = mvEfficientFrontier(mu=mean, cov=est, nPf=100, cov_orig=sample)
    ret = ef['expReturn']
    vol = ef['Volatility']
    returns.append(ret)
    volatilities.append(vol)
    
    # Plot each frontier:
    if i == (len(cov_list) - 1):
        black_line, = plt.plot(volatilities[i], returns[i], color='black', label="RERCE", linewidth=1.5)
    else:
        grey_line, = plt.plot(volatilities[i], returns[i], color='lightgrey', label="ERCE", antialiased=False, linewidth=0.9)

plt.legend(handles=[black_line, grey_line], labels=["RERCE", "ERCE"], loc="lower right")

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.show()


# Table 3: In-Sample Sharpe Ratios of MVPs and IPPs
# Compute annualized covariance matrix estimates  
sample = np.cov(return_data_subset, rowvar=False)*12
ERCE = robust_eigenvalue_modifier(return_data_subset_array, grand_avg = True)*12
RERCE = resampled_eigenvalue_modifier(return_data_subset_array,50)['cov_est']*12
LSCE = LedoitWolf().fit(return_data_subset).covariance_*12
NLSCE = LW_nonlin_shrink(return_data_subset)*12
cov_est = [sample, ERCE, RERCE, LSCE, NLSCE]

# Compute in-sample Sharpe ratios of MVPs
# Compute MVPs
w_MVP = np.zeros((11,6))
vol = np.zeros(6)
w_MVP[:,-1] = 1/11
for j, est in enumerate(cov_est):
    # Compute weight vector
    w_MVP[:,j] = MV_optimization(est)
    # Compute in-sample risk
    vol[j] = np.sqrt(w_MVP[:,j].T @ sample @ w_MVP[:,j])

risk_ew = np.sqrt(w_MVP[:,-1].T @ sample @ w_MVP[:,-1])
vol[-1] = risk_ew

exp_return = w_MVP.T @ mean
excess_return = exp_return - rf
sharpe_ratio = excess_return / vol


# Compute in-sample Sharpe ratios of IPPs
# Compute efficient frontiers
result_trad = mvEfficientFrontier(mu=mean, cov=sample,nPf=350)
result_LSCE = mvEfficientFrontier(mu=mean, cov=LSCE,nPf=100, cov_orig = sample)
result_NLSCE = mvEfficientFrontier(mu=mean, cov=NLSCE,nPf=100, cov_orig = sample)
result_ERCE = mvEfficientFrontier(mu=mean, cov=ERCE,nPf=100, cov_orig = sample)
result_RERCE = mvEfficientFrontier(mu=mean, cov=RERCE,nPf=350, cov_orig = sample)
frontiers = [result_trad, result_ERCE, result_RERCE, result_LSCE, result_NLSCE]

# Compute IPPs
w_IPP=np.zeros((11,6))
w_IPP[:,-1] = 1/11
vol_IPP = np.zeros(6)
for j, ef in enumerate(frontiers):
        # Compute weight vector
        matching_index = np.abs(ef['Volatility'] - risk_ew).argmin()
        matching_pf = ef['Portfolios'][matching_index,:]
        w_IPP[:,j] = matching_pf
        # Compute in-sample risk
        vol_IPP[j] = np.sqrt(w_IPP[:,j].T @ sample @ w_IPP[:,j])
vol_IPP[-1] = risk_ew

exp_return_IPP = w_IPP.T @ mean
excess_return_IPP = exp_return_IPP - rf
sharpe_ratio_IPP = excess_return_IPP / vol_IPP

# Store Sharpe ratios
sharpe_ratio_df = pd.DataFrame({
    'Methods': ['sample', 'ERCE', 'RERCE', 'LSCE', 'NLSCE', 'EWP'],
    'MVP': sharpe_ratio,
    'IPP': sharpe_ratio_IPP
})
#sharpe_ratio_df.to_latex("sharpe_ratio.tex", index=False, float_format="%.2f")


# Figure 3: In-Sample Comparison of MV-Efficient Portfolios from Covariance Estimators
# Risk and return of equal weighted portfolio
n_assets = return_data_subset.shape[1]
ewp = np.full(n_assets, 1 / n_assets)
ewp_risk = np.sqrt(ewp@sample@ewp)
ewp_return = np.dot(ewp,mean)

# Risk and return of individual assets
risk_asset = np.sqrt(np.diag(sample))
return_asset = mean

# Create the figure
plt.figure(figsize=(10, 6))

cov = ['Sample', 'ERCE', 'RERCE', 'LSCE', 'NLSCE']
colors = ['blue', 'orange', 'green', 'red', 'purple']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot the different Efficient Frontiers with different colors
plt.plot(result_trad['Volatility'], result_trad['expReturn'], label=cov[0], color=colors[0], linestyle='-')
plt.plot(result_ERCE['Volatility'], result_ERCE['expReturn'], label=cov[1], color=colors[1], linestyle='-')
plt.plot(result_RERCE['Volatility'], result_RERCE['expReturn'], label=cov[2], color=colors[2], linestyle='-')
plt.plot(result_LSCE['Volatility'], result_LSCE['expReturn'], label=cov[3], color=colors[3], linestyle='-')
plt.plot(result_NLSCE['Volatility'], result_NLSCE['expReturn'], label=cov[4], color=colors[4], linestyle='-')

# Add the EWP as a point
plt.scatter(ewp_risk, ewp_return, color='red', label='EWP', s=25, zorder=5)
plt.text(ewp_risk, ewp_return, 'EWP', fontsize=12, ha='right', va='bottom')

# Add the assets as points
plt.scatter(risk_asset, return_asset, color='black', label='Assets', s=25, zorder=5)
for i, name in enumerate(tickers):
    plt.text(risk_asset[i], return_asset[i], name, fontsize=12, ha='right', va='bottom')
plt.axvline(x=ewp_risk, color='black', linestyle=':')#, label=f'EWP Volatility = {ewp_risk:.2f}')

# Add grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Set axis labels and title
plt.xlabel('Volatility')
plt.ylabel('Expected Return')

plt.legend(loc='lower right', fontsize=12, title_fontsize=14, frameon=True, fancybox=True)
plt.tight_layout()
plt.show()


# Figure 4: Composition Maps Derived from the Sample Estimator and RERCE
# Resize font of legend and axes labels for composition maps
plt.rcParams.update({'font.size': 20})
plot_composition_map(result_trad['Portfolios'], result_trad['Volatility'], tickers, legend = False)
plot_composition_map(result_RERCE['Portfolios'], result_RERCE['Volatility'], tickers)
