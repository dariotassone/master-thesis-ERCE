# -*- coding: utf-8 -*-
"""
@author: Dario Tassone
"""
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import itertools
from sklearn.covariance import LedoitWolf
from scipy.linalg import toeplitz
from matplotlib import cm  # Import the colormap module
import copy
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
import os


def plot_composition_map(weight_matrix, volatility, tickers, legend=True):
    """
    This function computes a resampled efficient frontier based on a normality assumption of asset returns.
    Efficient frontiers are aggregated by averaging over simulated weights.

    Parameters
    ----------
    mu : numpy.ndarray
        Mean vector of the underlying assets.
    cov : numpy.ndarray
        Covariance matrix of the underlying assets.
    nPf : int
        Number of portfolios on each simulated frontier.
    nSim : int
        Number of simulated efficient frontiers.
    nDP : int
        Number of simulated data points (returns) for each asset.

    Returns
    -------
    dict
        A dictionary containing:
        - sim_frontiers : pd.DataFrame
            Efficient frontier of all simulations.
        - avg_weights : numpy.ndarray
            Averaged portfolio weights across simulations.
        - res_return : numpy.ndarray
            Returns of portfolios on the resampled frontier.
        - res_risk : numpy.ndarray
            Volatility of portfolios on the resampled frontier.
    """

    volatility = np.array(volatility)
    _, unique_indices = np.unique(volatility, return_index=True)
    sorted_indices = np.sort(unique_indices)
    sorted_indices = sorted_indices.astype(int)
    volatility = volatility[sorted_indices]

    weight_matrix = weight_matrix[sorted_indices]
    volatility = volatility.tolist()
    
    weight_matrix = np.transpose(weight_matrix)
    n_assets = weight_matrix.shape[0]
    n_portfolios = weight_matrix.shape[1]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Stacking bar plot: Loop through each row in efficientWeights (each asset)
    bottom = np.zeros(n_portfolios)  # Start stacking from the bottom
    colors = plt.cm.rainbow(np.linspace(0, 1, n_assets))  # Generate 20 colors (one per asset)

    # Plot each row as a layer in the stacked bar chart
    for i in range(n_assets):
            ax.bar(range(n_portfolios), weight_matrix[i], bottom=bottom, color=colors[i], edgecolor=colors[i], label=tickers[i])
            bottom += weight_matrix[i]  # Stack the next bar on top

    # Set x and y labels
    #plt.xlabel('Risk (Volatility)', fontsize=16)
    #plt.ylabel('Weights', fontsize=16)
    plt.rc('axes', titlesize=20)   # Title font size
    #plt.rc('axes', labelsize=20)   # X and Y label font size
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Weights')


    # Set custom x-axis labels (volatility)
    x = np.arange(n_portfolios)
    interval = n_portfolios // n_assets + 4
    vol_rounded = np.round(volatility,3)
    plt.xticks(ticks=x[::interval], labels=vol_rounded[::interval], rotation=70)

    # Add a legend (optional, since you have 20 assets, you can limit or modify it)
    if legend:
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1, title="Assets")

    # Show the plot
    plt.tight_layout()
    plt.show()



def downside_deviation(row, target=0):
    """
    This function computes the downside deviation of a given set of returns relative to a target return.
        
    Parameters
    ----------
    row : np.ndarray or pandas.Series
        Array or series of returns for which the downside deviation is to be calculated.
    target : float, optional
        The target return to compare against. Default is 0.
    
    Returns
    -------
    float
        Resulting performance gauge.
    """
    # Calculate deviations from the target
    downside = (row[row < target] - target) ** 2
    # Compute the mean of the squared negative deviations
    mean_squared_downside = downside.sum() / len(row)
    # Take the square root to get the downside deviation
    return np.sqrt(mean_squared_downside) 


    
def calc_cvar(row_returns, confidence_level, num_bootstrap_samples,rng=10):
    """
    This function calculates the Conditional Value at Risk (CVaR) for a given set of returns.
    It resamples the input returns using bootstrap techniques and computes the average CVaR across the resampled datasets.
    
    Parameters
    ----------
    row_returns : np.ndarray
        Array of historical or simulated returns.
    confidence_level : float
        The confidence level for VaR and CVaR calculations.
    num_bootstrap_samples : int
        The number of bootstrap samples to generate.
    
    Returns
    -------
    float
        The negative average CVaR value (positive number) calculated over the bootstrap samples.
"""

    rng_state = np.random.get_state()
    np.random.seed(rng)
    # Pre-allocate empty list
    bootstrap_cvars = []
    for _ in range(num_bootstrap_samples):
        # Generate a bootstrapped sample
        bootstrap_sample = np.random.choice(row_returns, size=len(row_returns), replace=True)
        
        # Calculate VaR95%
        var95 = np.percentile(bootstrap_sample, (1 - confidence_level) * 100)
        
        # Calculate CVaR95%
        cvar95 = np.mean(bootstrap_sample[bootstrap_sample <= var95])
        
        # Store the value
        bootstrap_cvars.append(cvar95)

    np.random.set_state(rng_state)
    # Average CVaR over all bootstrap samples
    return -np.mean(bootstrap_cvars)

def oos_performance_analysis(return_data, rf_df, estimation_window, holding_window, nMet, short_constraint = True, MVP=True):
    """
    This function implements the empirical performance analysis of portfolios derived by the sample covariance estimator, 
    RERCE, ERCE, LSCE, and NLSCE. The equal-weighted portfolio is employed as a benchmark. The function uses a rolling 
    window with variable estimation and holding windows.
    Robustness, diversification, risk and return are measured.
    Robustness and diversification are measured directly after rebalancing.
    Risk and return gauges are computed based on monthly portfolio returns.
    

    Parameters
    ----------
    return_data : pd.DataFrame
        DataFrame containing historical/simulated asset returns.
    rf_df : pd.DataFrame
        DataFrame containing risk-free rates for the corresponding periods.
    estimation_window : int
        The number of observations (in months) used in each estimation window for covariance matrix calculation.
    holding_window : int
        The number of periods (in months) for which the portfolio weights are held constant after estimation.
    nMet : int
        The number of covariance estimation methods being evaluated (e.g., sample, RERCE, ERCE, LSCE, NLSCE).
    short_constraint : bool, optional
        If True, enforces no short-selling in the portfolio. Default is True.
    MVP : bool, optional
        If True, includes the computation of the Minimum Variance Portfolio (MVP). Default is True.
        Else, it computes the Intersection Point Portfolio (IPP), as defined in Subsection 5.2.2

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - df_structure : pd.DataFrame
            DataFrame summarizing the structure of the covariance estimates and derived portfolio allocations
            contains information about robustness, diversification and realized risk.
        - df_performance : pd.DataFrame
            DataFrame summarizing the performance metrics of the portfolios in terms of Sharpe ratio and CAGR.
        - w : np.ndarray
            Array of portfolio weights derived during the analysis.
        - portfolio_gross_return_cum : pd.Series
            Cumulative gross returns of the portfolio.
        - portfolio_gross_return_cum_tc : pd.Series
            Cumulative gross returns of the portfolio, adjusted for transaction costs.
    """
    # Number of assets
    k = return_data.shape[1]
    
    # Investment horizon in years (total number of years - length of first estimation window)
    investment_period = 30 - estimation_window/12
    
    # Number of rebalancing dates per year
    rebalancing_per_year = 12/holding_window
    
    # Number of rebalancing dates
    p = int(investment_period*rebalancing_per_year)
    
    # Pre-allocation of objects filled within the analysis    
    # Robustness of covariance matrix and MVP
    cond_number_mat = np.zeros((nMet, p))
    WD = []
    turnover_mat = np.zeros((nMet+1, p-1))
    
    # Diversification of MVP
    hhi_mat = np.zeros((nMet+1, p))
    share_short_position_mat = np.zeros((nMet+1,p))
    
    # Three-dimensional weight matrix:
    # For each tested method, a (k x p) matrix of weights is created
    # The rows (k) represent the number of assets
    # The columns (p) indicate the rebalancing dates
    w = np.zeros((k, p, nMet+1))
    # Last dimension is the weight matrix of the EWP
    w[:,:,5]  = 1/k
    
    # Two-dimensional matrix overwritten iteratively
    w_current = np.zeros((k,nMet+1))
    w_current[:,nMet] = 1/k
    
    # Monthly/Periodic Return objects
    # portfolio_gross_return will be filled with gross returns from one to the next rebalancing date
    portfolio_gross_return = np.zeros((nMet+1,p+1))
    portfolio_gross_return[:,0] = 1

    # Monthly gross return objects
    returns_monthly_gross = np.zeros((nMet+1, 30*12-estimation_window + 1))
    returns_monthly_gross[:,0] = 1
    returns_monthly_gross_tc = copy.deepcopy(returns_monthly_gross)

    # Transaction cost factor
    tc = 0.005

    # Iterate through rebalancing dates
    for i in range(0,p):
        
        # At each rebalancing date, define estimation and holding window
        estimation_data =  return_data.iloc[(holding_window*i):(estimation_window+i*holding_window),:]
        estimation_data_array = np.array(estimation_data)
        holding_data = return_data.iloc[(estimation_window + holding_window*i):(estimation_window + holding_window*(i+1)),:]
        holding_data_rf = rf_df.iloc[(estimation_window + holding_window*i):(estimation_window + holding_window*(i+1)),:]
                
        # Boundaries of estimation window
        #print(holding_window*i)
        #print(estimation_window+i*holding_window)
        
        # Boundaries of holding window
        #print(estimation_window + holding_window*i)
        #print(estimation_window + holding_window*(i+1))
        
        # Monthly gross returns
        gross_return = holding_data + 1
        
        # Compute gross return over holding window for each asset
        holding_return = gross_return.apply(lambda x: x.prod(), axis=0)
        
            
        # Compute covariance matrix estimates based on estimation window 
        sample = np.cov(estimation_data, rowvar=False)
        ERCE = robust_eigenvalue_modifier(estimation_data_array, grand_avg = True)
        RERCE = resampled_eigenvalue_modifier(X=estimation_data_array, nSim=100, grand_avg = True, m_rel= None, rng=10, rng_flag=True)['cov_est']
        LSCE = LedoitWolf().fit(estimation_data).covariance_
        NLSCE = LW_nonlin_shrink(estimation_data)
        
        # Replace the singular sample estimate with a placeholder. Results will later be overwritten with NAs if n < k.
        if k>estimation_window:
            sample = RERCE
        
        cov_est = [sample, ERCE, RERCE, LSCE, NLSCE]

        # If matching equal-weighted portfolio (not MVP!) is demanded 
        if not MVP:
            # Equal-weighted portfolio
            ewp = np.full(k,1/k)
            # Risk of equal weighted portfolio (based on sample estimate!)
            risk_ew = np.sqrt(ewp@sample@ewp)
            # Mean vector of assets (Note: only relevant for matching portfolio. Not for MVP)
            mean = np.array(estimation_data.mean())

        # Iterate through covariance estimates based on simulated data
        for j, est in enumerate(cov_est):
            if MVP:
                # Compute MVP allocation based on covariance estimate est
                w_current[:,j] = MV_optimization(est)
            else:
                # Compute efficient frontier
                ef = mvEfficientFrontier(mu=mean, cov=est,nPf=20, short_constraint=short_constraint)
                # Find portfolio with most similar risk level compared to EWP
                matching_index = np.abs(ef['Volatility'] - risk_ew).argmin()
                matching_pf = ef['Portfolios'][matching_index,:]
                w_current[:,j] = matching_pf
            
            # Store current weight vector
            w[:,i,j] = w_current[:,j]
            
            # Fill performance metrics
            hhi_mat[j,i] = hhi(w_current[:,j])
            cond_number_mat[j,i] = np.linalg.cond(est)
    
        # Update portfolio gross returns (incl. TC)
        portfolio_gross_return[:,i+1] = np.dot(w_current.T, holding_return)
        
        # Compute monthly returns over holding period based on allocations at preceding rebalancing date
        weights_init = w_current.copy()
        for h in range(holding_window):
            # For each strategy, compute portfolio return in month h of holding window
            r = np.dot(holding_data.iloc[h,:].values,weights_init)
            returns_monthly_gross[:,h+i*(holding_window)+1] = 1 + r
            returns_monthly_gross_tc[:,h+i*(holding_window)+1] = 1 + r
            # For returns_monthly_gross_tc, the last month of each holding window will be adjusted for transaction costs
            
            # Compute weights adjusted for asset returns realized in month h
            weights_init = weights_init*(1+holding_data.iloc[h,:].values[:,np.newaxis])
            weights_init = weights_init / np.sum(weights_init, axis=0)                        
            
        # Compute turnover (from second rebalancing date) using weights adjusted
        # for realized returns over the preceding holding period
        if i != 0:
            w_diff = np.abs(w_current-adjusted_w)
            # Compute the column sum (sum of asset-wise weight differences )
            col_sums = np.sum(w_diff, axis=0)
            turnover_mat[:,i-1] = col_sums            
            transaction_cost_factor = 1 - col_sums*tc
            
            # Adjust returns for transaction costs
            returns_monthly_gross_tc[:,holding_window+i*(holding_window)] = returns_monthly_gross_tc[:,holding_window+i*(holding_window)] * transaction_cost_factor
        
        # Update adjusted weight matrix for next iteration
        if i != (p-1):
            holding_return_array = np.array(holding_return)
            # Compute weight at the end of the current holding window
            adjusted_w = w_current*holding_return_array[:, np.newaxis]  / portfolio_gross_return[:,i+1].T
    
    
        # Proportion of short positions
        short_flag = w_current<0
        # Absolute value of sum of negative weights
        abs_sum_negative = np.abs(np.sum(w_current*short_flag, axis=0))
        # Sum of absolute values of all weights
        sum_abs_total = np.sum(np.abs(w_current), axis=0)
        # Compute gauge for each period
        share_short_position_mat[:,i] = abs_sum_negative / sum_abs_total
    
    
    # Share of short position
    short = np.mean(share_short_position_mat, axis=1)
    
    # Turnover
    turnover = np.mean(turnover_mat, axis=1)

    # Condition
    condition_number = np.mean(cond_number_mat, axis=1)
    # Account for EWP
    condition_number = np.append(condition_number, np.nan)
    
    # HHI
    w_current[:,nMet] = 1/k
    hhi_mat[nMet,:] = hhi(w_current[:,nMet])
    HHI = np.mean(hhi_mat, axis=1)
    
    
    
    # Loop through weight matrix of each layer
    for i in range(w.shape[2]):
        # Extract the weight matrix of the i-th layer
        layer = w[:, :, i]
        # Compute the standard deviation for each row (weight deviation per asset over time)
        row_std = np.std(layer, axis=1)
        # Compute the average of the row standard deviations
        avg_std = np.mean(row_std)
        # Append the result to the list
        WD.append(avg_std)
    # Convert to array
    WD = np.array(WD)
    
    
    # Compute portfolio returns cumulated over months
    returns_monthly_cum = np.cumprod(returns_monthly_gross, axis=1)
    returns_monthly_cum_tc = np.cumprod(returns_monthly_gross_tc, axis=1)
    
    # Compute net portfolio returns; remove first element
    returns_monthly = returns_monthly_gross[:,1:] - 1
    returns_monthly_tc = returns_monthly_gross_tc[:,1:] - 1

    # Compute monthly excess returns
    rf_subset = rf_df.iloc[estimation_window:].values
    returns_monthly_excess = returns_monthly - rf_subset.T/12
    returns_monthly_excess_tc = returns_monthly_tc - rf_subset.T/12


    # Risk indicators
    target_return = 0
    portfolio_excess_return_df = pd.DataFrame(returns_monthly_excess)
    # Apply the function row-wise to the return data frame
    DD = portfolio_excess_return_df.apply(downside_deviation, axis=1, target=target_return)*np.sqrt(12)
    
    # Compute maximum drawdown
    MDD = []
    for m in returns_monthly_cum:
        running_max = np.maximum.accumulate(m)
        drawdowns = (running_max - m) / running_max
        MDD.append(np.max(drawdowns))
        
    MDD = np.array(MDD)
    

    # Compute CVaR95%
    cvar = []
    # Iterate through annualized net returns of each method
    for m in returns_monthly:
        cvar95 = calc_cvar(m, confidence_level=0.95, num_bootstrap_samples=1000)
        cvar.append(cvar95)
    cvar = np.array(cvar)
    
    
    # Risk-adjusted Performance Measures
    # CAGR (w/o transaction cost)
    terminal_val = returns_monthly_cum[:, -1].flatten()
    CAGR = np.full_like(terminal_val, np.nan, dtype=np.float64)
    mask_pos = terminal_val >= 0 # only meaningful for wealth growth
    # Annual basis:
    CAGR[mask_pos] = terminal_val[mask_pos]**(1 / (30 - estimation_window / 12)) - 1

    # CAGR (adjusted for transaction cost)
    terminal_val_tc = returns_monthly_cum_tc[:, -1].flatten()
    CAGR_tc = np.full_like(terminal_val_tc, np.nan, dtype=np.float64)
    mask_pos_tc = terminal_val_tc >= 0
    CAGR_tc[mask_pos_tc] = terminal_val_tc[mask_pos_tc]**(1 / (30 - estimation_window / 12)) - 1

    # Portfolio volatility (annualized)
    portfolio_realized_variance = np.var(returns_monthly_excess, axis=1, ddof=1)
    portfolio_realized_vol = np.sqrt(portfolio_realized_variance)*np.sqrt(12)

    # Portfolio volatility (annualized) of returns adjusted for transaction costs
    portfolio_realized_variance_tc = np.var(returns_monthly_excess_tc, axis=1, ddof=1)
    portfolio_realized_vol_tc = np.sqrt(portfolio_realized_variance_tc)*np.sqrt(12)

    # Portfolio mean return (annualized)
    portfolio_mean_return = np.mean(returns_monthly_excess, axis=1)*12
    portfolio_mean_return_tc = np.mean(returns_monthly_excess_tc, axis=1)*12

    # Sharpe ratio based on annualized components
    sharpe_ratio = portfolio_mean_return / portfolio_realized_vol
    sharpe_ratio_tc = portfolio_mean_return_tc / portfolio_realized_vol_tc


    
    method = ['sample', 'ERCE', 'RERCE', 'LSCE', 'NLSCE', 'EW']
    # Summarize results in data frame
    df_structure = pd.DataFrame({
        'n':estimation_window,
        'k':k,
        'Method': method,
        # Robustness (of MVP)
        'Condition Number': condition_number,
        'WD': WD,
        'Turnover': turnover,
        # Diversification (of MVP)
        'HHI': HHI,
        'Short': short,
        # Risk measures
        'Vol' : portfolio_realized_vol,
        'DSD': DD,
        'MDD': MDD,
        'CVaR95': cvar
    })
    
    df_performance = pd.DataFrame({
        'n':estimation_window,
        'k':k,
        'Method': method,
        # Performance
        'CAGR': CAGR,
        'Sharpe Ratio': sharpe_ratio,
        'Mean Return': portfolio_mean_return,
        'Vol' : portfolio_realized_vol,
        # Performance adjusted for transaction costs
        'CAGR TC': CAGR_tc,
        'Sharpe Ratio TC': sharpe_ratio_tc,
        'Mean Return TC': portfolio_mean_return_tc,
        'Vol TC': portfolio_realized_vol_tc
    })
    
    # Note: If k exceeds n, the inverse of the sample covariance matrix does not exist
    # Since the inverse is required to compute the MVP, 
    # Results are overwritten with NAs
    if k>=estimation_window:
        df_structure.iloc[0,3:] = np.nan
        df_performance.iloc[0,3:] = np.nan
        w[:,:,0] = np.nan
        returns_monthly_cum[0,:] = np.nan
        returns_monthly_cum_tc[0,:] = np.nan

    return {
            'df_structure': df_structure,
            'df_performance': df_performance,
            'w': w,
            'monthly_return_cum': returns_monthly_cum,
            'monthly_return_cum_tc': returns_monthly_cum_tc
        }
    
def mv_optimization_numerical(mu, cov, R, short_constraint = True, cov_orig=None):
    """
    This function numerically computes the weights of the efficient, mean-variance optimized portfolio.
    It minimizes portfolio variance for a given expected portfolio return.

    Args:
            mu (numpy.ndarray): mean vector of the underlying assets
            cov (numpy.ndarray): covariance matrix of the underlying assets
            R (float): expected portfolio return

    Returns: (dict)
            weight (numpy.ndarray): vector of optimal weights
            return (float): expected portfolio return
            risk (float): volatility of portfolio return
    """

    # Specify number of assets
    n_assets = len(mu)

    # Specify constraints - based on inputs required for solve_qp()
    # Linear weight constraints: weights sum to 1; weighted sum of returns equals specified expected return  
    A = np.vstack((mu, np.ones(n_assets)))
    b = np.array([R, 1])
    q = np.zeros(n_assets)

    # Short-sale constraint for all assets
    if short_constraint:
        lower_bound = np.zeros([n_assets])
        # Computation of weight vector
        weights = solve_qp(
                P = cov,
                q = q,
                G = None,
                h = None,
                A = A,
                b = b,
                lb = lower_bound,
                hb = None,
                solver='cvxopt'
                )
    else:
        weights = solve_qp(
                P = cov,
                q = q,
                G = None,
                h = None,
                A = A,
                b = b,
                lb=None,
                hb = None,
                solver='cvxopt'
                )

    # Computation of portfolio volatility
    if cov_orig is not None:
            cov = cov_orig
    
    risk = np.sqrt(weights@cov@weights)
    
    return {'weight': weights,
            'return': R,
            'risk': risk}
    
def MV_optimization(cov, mu = None, mvp = True):
    """
    This function computes the optimal portfolio allocation using the closed-form
    solutions derived from Markowitz' portfolio theory.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix of asset returns.
    mu : np.ndarray, optional
        Vector of expected asset returns.
    mvp : bool, optional
        If true, enables the computation of the Minimum Variance Portfolio (MVP, see A.1.1);
        otherwise, computes the tangency portfolio.

    Returns
    -------
    np.ndarray
        Vector of portfolio weights.
    """
    if mvp:
        num = np.linalg.inv(cov)@np.ones(len(cov))
        
    else:
        num = np.linalg.inv(cov)@mu
    den = np.sum(num)
    return num/den

def mvEfficientFrontier(mu, cov, nPf, short_constraint=True, flag_restock = False, cov_orig=None):
    """
    This function computes the efficient frontier for a set of assets.

    Parameters
    ----------
    mu : numpy.ndarray
        Mean vector of the underlying assets.
    cov : numpy.ndarray
        Covariance matrix of the underlying assets.
    nPf : int
        Number of portfolios on the efficient frontier.

    Returns
    -------
    dict
        A dictionary containing:
        - weight : numpy.ndarray
            Vector of optimal weights.
        - return : float
            Expected portfolio return.
        - risk : float
            Volatility of portfolio return.
    """

    # Determine minimum and maximum portfolio return
    # due to short-sale constraints, returns are bounded between min. and max. expected asset return
    minRet = mu.min()
    maxRet = mu.max()

    # Pre-allocate empty lists
    ret, risk, portfolios = [], [], []

    # Define sequence of equidistant portfolio returns 
    return_seq = np.linspace(minRet, maxRet, nPf)

    # Iterate through portfolio return sequence and determine respective optimal portfolio
    for iPf in return_seq:
            result = mv_optimization_numerical(mu=mu, cov=cov, R = iPf, short_constraint = short_constraint, cov_orig = cov_orig)
            ret.append(result['return'])
            risk.append(result['risk'])
            portfolios.append(result['weight'])
            
    # (Note: Up to this point, we obtain a sideways-facing parabola. Next, we extract the upper part.)
    
    # Pre-allocate empty lists
    eff_risk, eff_ret, eff_portfolios, eff_indices = [],[],[],[]
    # Iterate through computed portfolios and extract efficient subset
    for i in range(nPf):
            # Assume the current portfolio has the highest return for its risk
            flag_efficient = True
            # Filter out inefficient portfolios:
            for j in range(nPf):
                    if risk[j] <= risk[i] and ret[j] > ret[i]:
                            flag_efficient = False 
                            # Update index if a higher return is found for the same or lower risk
                            break
            if flag_efficient:
                    eff_risk.append(risk[i])
                    eff_ret.append(ret[i])
                    eff_portfolios.append(portfolios[i])
                    eff_indices.append(i)

    # Number of efficient portfolios (after filtering out portfolios by definition inefficient)
    currentNPF = len(eff_indices)
    
    # If too many portfolios have been filtered out, extend the set of portfolios (when desired)
    if flag_restock:
            if currentNPF < nPf:
            # Number of portfolios tb added
                    add_pf = nPf - currentNPF
                    
                    # Extend portfolios
                    eff_risk = [eff_risk[0]]*add_pf + eff_risk
                    eff_ret = [eff_ret[0]]*add_pf + eff_ret
                    eff_portfolios = [eff_portfolios[0]]*add_pf + eff_portfolios
    
    eff_portfolios = np.row_stack(eff_portfolios)
    
    return {'expReturn': eff_ret,
            'Volatility': eff_risk,
            'Portfolios': eff_portfolios}


def multivariate_t(mean, cov, size, df=5):
    # Dimension/ Number of assets
    d = len(mean)
    
    # Generate gamma distributed random variables
    g = np.random.gamma(df / 2., 2. / df, size=1)
    
    # Multivariate normal distribution
    z = np.random.multivariate_normal(np.zeros(d), cov, size=size)
    return mean + z / np.sqrt(g)[:, None]

def covariance_matrix_toeplitz(k, alpha):
    """
    This function computes a Toeplitz covariance matrix based on the given alpha and dimension k.

    Parameters
    ----------
    k : int
        The size of the matrix (number of assets).
    alpha : float
        The correlation parameter, with values like 0.0, 0.5, 0.75, or 0.95.
    
    Returns
    -------
    np.ndarray
        Toeplitz covariance matrix.
    """
    # Create the first column for the matrix
    first_col = [alpha ** abs(i) for i in range(k)]
    
    # Generate the Toeplitz covariance matrix
    cov_matrix = toeplitz(first_col)
    
    return cov_matrix

def robust_eigenvalue_modifier(X, grand_avg = True, m_rel = None):
    """
    This function computes the a covariance matrix estimator introduced in Abadir et al. (2014).
    
    Reference:
        Karim M Abadir, Walter Distaso, and Filip ˇ Zikeˇs. Design-free estimation of variance
        matrices. Journal of Econometrics, 181(2):165–180, 2014.
    
    Parameters
    ----------
    X : numpy.ndarray
        n x p matrix of asset returns (n observations of p assets).
    grand_avg : bool
        Flag enabling the computation of an averaged estimator based on specified sample
        splits of different sizes  (m in {0.2,0.4,0.6,0.8}).
    m_rel : float
        Specifies the share of data points used for the first part.
    
    Returns
    -------
    numpy.ndarray
        Covariance matrix estimate.
    """

    if grand_avg & (m_rel is not None):
        print("When grand_avg is set to True, individual sample splitting sizes cannot be specified manually. Set m_rel = None or grand_avg = True.")
    
    if grand_avg:
        # Pre-allocate list filled with estimates of different sample splits
        cov_ind = []
        # Iterate through pre-defined sample sizes
        for m_share in np.array([0.2,0.4,0.6,0.8]):
            
            # Divide data in two parts
            m_abs = round(m_share*X.shape[0])
            X_1 = X[0:m_abs,:]
            X_2 = X[m_abs:,:]
            
            # Compute covariance matrix eigenvectors of first part of data
            cov_1 = np.cov(X_1, bias=True,rowvar=False)
            eigenvalues_1, eigenvectors_1 = np.linalg.eigh(cov_1)
            P_1 = eigenvectors_1
            
            # Compute covariance matrix estimate of second part of data
            cov_2 = np.cov(X_2, bias=True,rowvar=False)
            
            # Estimate eigenvalues based on covariance structure of second part of data
            lambda_est = np.diag(np.diag(np.transpose(P_1)@cov_2@P_1))

            # Compute eigenvectors of entire data set
            cov_all = np.cov(X, bias=True,rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_all)
            P = eigenvectors
            # Calculate estimator
            cov_ind.append(P @ lambda_est @ np.transpose(P))
        
        # Compute final as element-wise average of individual estimators
        cov_est = np.sum(cov_ind, axis=0)/len(cov_ind)            
        
    if m_rel is not None:
        # Divide data in two parts
        m_abs = round(m_rel*X.shape[0])
        X_1 = X[0:m_abs,:]
        X_2 = X[m_abs:,:]
        
        # Compute covariance matrix eigenvectors of first part of data
        cov_1 = np.cov(X_1, bias=True,rowvar=False)
        eigenvalues_1, eigenvectors_1 = np.linalg.eigh(cov_1)
        P_1 = eigenvectors_1
        
        # Compute covariance matrix estimate of second part of data
        cov_2 = np.cov(X_2, bias=True,rowvar=False)
        
        # Estimate eigenvalues based on covariance structure of second part of data
        lambda_est = np.diag(np.diag(np.transpose(P_1)@cov_2@P_1))

        # Compute eigenvectors of entire data set
        cov_all = np.cov(X, bias=True,rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_all)
        P = eigenvectors
        # Calculate estimate
        cov_est = P @ lambda_est @ np.transpose(P)
    return cov_est


def resampled_eigenvalue_modifier(X, nSim, grand_avg = True, m_rel= None, rng=1, rng_flag=False):
    """
    This function computes a resampled modification of the estimator proposed
    by Abadir et al. (2014), based on the robust_eigenvalue_modifier() function.
    
    Parameters
    ----------
    X : numpy.ndarray
        n x p matrix of asset returns (n observations of p assets).
    nSim : int
        Number of resampling iterations.
    grand_avg : bool
        Flag enabling the computation of an averaged estimator based on specified sample
        splits of different sizes  (m in {0.2,0.4,0.6,0.8}).
    m_rel : float
        Specifies the share of data points used for the first part.
    

    Returns
    -------
    numpy.ndarray
        Resampled covariance matrix estimate.
    """
    sim_estimator_list = []
    if rng_flag:
        np.random.seed(rng)
    for i in range(0,nSim):
        X_bootstrap = X.copy()
        np.random.shuffle(X_bootstrap)
        sim_estimator = robust_eigenvalue_modifier(X_bootstrap, grand_avg = grand_avg, m_rel = m_rel)
        sim_estimator_list.append(sim_estimator)
    cov_est = np.sum(sim_estimator_list, axis=0)/len(sim_estimator_list)
    return {"cov_est": cov_est, "cov_est_sim_list": sim_estimator_list}


def LW_nonlin_shrink(x):
    """
    This function implements the non-linear shrinkage estimator of Ledoit and Wolf (2020).
    Reference:
    Olivier Ledoit and Michael Wolf. Analytical nonlinear shrinkage of large-dimensional
    covariance matrices. The Annals of Statistics, 48(5):3043–3065, 2020.

    Parameters
    ----------
    x : numpy.ndarray
        Data matrix whose columns represent different assets and rows represent observations.

    Returns
    -------
    np.ndarray
        The resulting covariance matrix estimate.
    """
    # Transpose and specify dimensions
    x = x.T
    p = x.shape[0]
    n = x.shape[1]

    #Compute sample estimate
    sampleC = np.cov(x, rowvar=True)
    
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(sampleC)
    u = eigvecs
    lambda_vals = eigvals

    # Case 1: number of observations is geq number of assets
    if p <= n:
        lambda_vals = lambda_vals[max(0, p - n):p]
        L = np.tile(lambda_vals, (min(p, n), 1)).T
        
        h = n**(-1/3)
        H = h * L.T
        
        x = (L - L.T) / H
        ftilde = (3 / (4 * np.sqrt(5))) * np.mean(np.maximum(1 - x**2 / 5, 0) / H, axis=1)
        
        Hftemp = (-3 / (10 * np.pi)) * x + (3 / (4 * np.sqrt(5) * np.pi)) * (1 - x**2 / 5) * \
                    np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
        
        Hftemp[np.abs(x) == np.sqrt(5)] = (-3 / (10 * np.pi)) * x[np.abs(x) == np.sqrt(5)]
        
        Hftilde = np.mean(Hftemp / H, axis=1)
        
        dtilde = lambda_vals / ((np.pi * (p / n) * lambda_vals * ftilde)**2 +
                                (1 - (p / n) - np.pi * (p / n) * lambda_vals * Hftilde)**2)
    
    # Case 2: number of assets exceeds number of obs
    else:
        lambda_vals = lambda_vals[max(0, p - n + 1):p]
        L = np.tile(lambda_vals, (min(p, n - 1), 1)).T

        h = n**(-1/3)
        H = h * L.T

        x = (L - L.T) / H

        ftilde = (3 / (4 * np.sqrt(5))) * np.mean(np.maximum(1 - x**2 / 5, 0) / H, axis=1)

        Hftemp = (-3 / (10 * np.pi)) * x + (3 / (4 * np.sqrt(5) * np.pi)) * (1 - x**2 / 5) * \
                    np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))

        Hftemp[np.abs(x) == np.sqrt(5)] = (-3 / (10 * np.pi)) * x[np.abs(x) == np.sqrt(5)]

        Hftilde = np.mean(Hftemp / H, axis=1)

        Hftilde0 = (1 / np.pi) * (3 / (10 * h**2) + (3 / (4 * np.sqrt(5) * h)) *
                        (1 - 1 / (5 * h**2)) * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))) * \
                        np.mean(1 / lambda_vals)

        dtilde0 = 1 / (np.pi * (p - n) / n * Hftilde0)
        dtilde1 = lambda_vals / (np.pi**2 * lambda_vals**2 * (ftilde**2 + Hftilde**2))

        dtilde = np.concatenate([dtilde0 * np.ones(p - n + 1), dtilde1])
    
    cov_est = u @ np.diag(dtilde) @ u.T
    return cov_est


def vech(matrix):
    """
    This function extracts the lower triangular part of a matrix,
    including the diagonal.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix whose lower triangular part is extracted.

    Returns
    -------
    np.ndarray
        Array of elements from the lower triangular part of the matrix, including the diagonal.
    """
    return matrix[np.tril_indices(matrix.shape[0])]

def error_criterion(mat, mat_est, l=2, scale=100):
    """
    This function computes a criterion to measure the precision of an estimated
    and a population matrix based on the generalized l-norm. The criterion is
    designed for symmetric matrices. By using vech(), it is ensured that the
    deviation of each pair is considered once.
    
    Parameters
    ----------
    
    mat : numpy.ndarray
        Population matrix (symmetric).
    mat_est : numpy.ndarray
        Estimated matrix (symmetric).
    l : int
        Order of the l-norm; Euclidean norm is Default.

    Returns
    -------
    float
        Error criterion measuring the precision of the estimator `mat_est`.
    """
    diff = mat-mat_est
    vech_diff = vech(diff)
    norm = np.linalg.norm(vech_diff, ord=l)
    
    return norm * scale


def hhi(weights, scale=10000, standardized = True):
    """
    This function computes the Herfindahl-Hirschman Index (HHI) for a given list of portfolio weights.
    It accounts for short positions
    
    Parameters
    ----------
    weights : np.ndarray
        Weights of estimated portfolio allocation.
    standardized : bool
        Flag specifying whether the standardized HHI (bounded between 0 and 1) is computed.

    Returns
    -------
    hhi : float
        Resulting performance gauge
    """
    abs_weights = np.abs(weights)

    # Normalize the weights
    normalized_weights = abs_weights / np.sum(abs_weights)

    # Compute the HHI as the sum of squared normalized weights
    hhi = np.sum(normalized_weights ** 2)
    
    # Standardize the HHI if the flag is set
    if standardized:
        k = len(weights)
        return (hhi - 1 / k) / (1 - 1 / k)*scale
    
    return hhi*scale


def weight_MAD(weights, opt_weights = None):
    """
    This function computes the mean absolute deviation of an estimated allocation from its population counterpart.
    If no optimal allocation is provided, the mean absolute deviation from the equal-weighted portfolio is computed. 
    
    Parameters
    ----------
    weights : np.ndarray
        Weights of estimated portfolio allocation.
    opt_weights : np.ndarray
        Weights of respective optimal portfolio allocation (e.g. based on population moments).

    Returns
    -------
    mad : float
        Resulting performance gauge
    """
    n = len(weights)
    if opt_weights is not None:
        mad = 1/n * sum(abs(weights - opt_weights))
    else:
        mad = 1/n * sum(abs(weights - 1/n))
    return mad


def simulation_based_performance(cov, k, n, nSim, nMet, dist_function= np.random.multivariate_normal, rng=1):
    """
    This function evaluates the performance of the following covariance matrix estimation methods 
    (and their variations) using Monte Carlo simulations:
    - Sample covariance matrix
    - ERCE
    - RERCE
    - LSCE
    - NLSCE
    
    Parameters
    ----------
    cov : np.ndarray
        The true covariance matrix of dimensions (k, k).
    k : int
        The number of assets.
    n : int
        The number of observations per simulation.
    nSim : int
        The number of simulation runs.
    nMet : int
        The number of covariance estimation methods being evaluated.
    dist_function : callable
        A function to generate random samples with the given covariance matrix.
        Default is a multivariate normal distribution.

    Returns
    -------
    results_df : pd.DataFrame
        A DataFrame summarizing the performance of each method across various evaluation metrics:
        Precision of covariance matrix and MVP
            - ERC: Error Criterion measuring the difference to the true covariance matrix.
            - PRIAL: Percentage Reduction In Average Loss (relative to the sample estimate).
            - MVPD: Minimum Variance Portfolio Deviation of estimated weights from optimal weights.
        Robustness of covariance matrix and MVP
            - Condition Number: Condition number of the covariance matrix estimates.
            - WD: Weight deviation - Mean standard deviation of weights across simulations.
        Diversification of MVP
            - HHI: Herfindahl-Hirschman Index for weight concentration.
        Asset Discrimination Ability
            - rho, rho_sign: Correlation and statistical significance of relationship of weights and asset covariances.
    """
    ### Pre-allocation of objects filled within simulation process
    # Precision of covariance matrix and MVP
    ERC = np.zeros((nSim,nMet))
    MVPD = np.zeros((nSim,nMet))
    
    # Robustness of covariance matrix and MVP
    cond_number = np.zeros((nSim,nMet))
    
    # Diversification of MVP
    hhi_mat = np.zeros((nSim,nMet))

    # List of weight matrices for each approach (covariance matrix estimate)
    w = [np.zeros((k, nSim)) for _ in range(nMet)]
    
    # Population MVP
    w_opt = MV_optimization(cov)
    
    np.random.seed(rng)
    # Iteration through nSim simulation runs
    for i in range(0,nSim):
        # Generate random data
        random = dist_function(mean=np.zeros(k), cov=cov, size=n)
        # Compute covariance estimates
        sample = np.cov(random, rowvar=False)
        ERCE = robust_eigenvalue_modifier(random, grand_avg = True)
        RERCE = resampled_eigenvalue_modifier(X=random, nSim=10, grand_avg = True, m_rel= None, rng=1, rng_flag=False)['cov_est']
        LSCE = LedoitWolf().fit(random).covariance_
        NLSCE = LW_nonlin_shrink(random)
        cov_est = [sample, ERCE, RERCE, LSCE, NLSCE]

        # Iteration through covariance estimates based on simulated data
        for j, est in enumerate(cov_est):
            
            # Compute estimate of MVP based on respective covariance estimate
            w[j][:,i] = MV_optimization(est)
            
            # Fill performance metrics
            ERC[i, j] = error_criterion(cov, est)
            hhi_mat[i,j] = hhi(w[j][:,i])
            MVPD[i,j] = weight_MAD(weights=w[j][:,i], opt_weights=w_opt)
            cond_number[i,j] = np.linalg.cond(est)
            
    # Calculate the mean standard deviation of weights per asset across simulations
    WD = np.array([np.std(w[i], axis=1).mean() for i in range(nMet)])
    
    # Calculate verage covariance per asset
    avg_cov = np.mean(cov,axis=1)
    
    # Compute correlation of average covariances and average weights per asset
    rho = np.array([pearsonr(avg_cov, np.mean(w[i], axis=1))[0] for i in range(nMet)])
    rho_sign = np.array([pearsonr(avg_cov, np.mean(w[i], axis=1))[1] for i in range(nMet)])
    
    # For each method, compute metric as average value across simulations
    cond, HHI, MVPD, ERC = [np.mean(metric, axis=0) for metric in [cond_number, hhi_mat, MVPD, ERC]]
   
    # Percentage reduction in average loss (relative to sample estimate)
    PRIAL = ((ERC[0] - ERC) / ERC[0])

    # Summarize results in data frame
    result = pd.DataFrame({
        'Method': np.array(['Sample', 'ERCE', 'RERCE','LSCE', 'NLSCE']),
        # Precision
        'ERC':ERC,
        'PRIAL': PRIAL,
        'MVPD':MVPD,
        # Robustness
        'Condition Number': cond,
        #'EV Deviation': eigenvalue_dev,
        'WD': WD,
        # Diversification (of MVP)
        'HHI': HHI,
        # Discrimination (of MVP)
        'corr':rho,
        'corr_sign':rho_sign
    })
    
    # Note: If k exceeds n, the sample covariance matrix does not exist - set values to nan
    if k>=n:
        result.iloc[0,1:] = np.nan
        result['PRIAL'] = np.nan
    
    return result


def simulation_result_generator(cov_function, k_values, n_values, nSim, nMet, dist_function, alpha=None, rng=1): 
    """
    Generates simulation-based performance results for various combinations of 'k' (number of assets) and 'n' (number of returns) using a specified covariance function 
    and distribution function.

    Parameters
    ----------
    cov_function : function
        Function that generates the true covariance matrix.
    k_values : list
        List of different numbers of random variables (assets) to be tested.
    n_values : list
        List of different numbers of realizations (observations/periods) per simulation iteration to be tested.
    nSim : int
        Number of simulation runs.
    nMet : int
        Number of covariance estimators to be tested.
    dist_function : function
        Distribution of drawn random values (np.random.multivariate_normal() or e.g. multivariate_t()).
    alpha : float
        Specifies elements of Toeplitz matrix; required when cov_function == covariance_matrix_toeplitz.

    Returns
    -------
    results_df : pd.DataFrame
        Data frame of performance metrics for each (n,k) combination.

    """
    results_dict = {}
    
    if cov_function == covariance_matrix_toeplitz and alpha is None:
        print("Error: Using a Toeplitz design as the true covariance matrix requires specifying the value of alpha.")
        return
    
    # Iterate through all combinations of elements in k_values and n_values
    for k, n in itertools.product(k_values, n_values):
        
        # Generate covariance matrix for the given k
        cov = cov_function(k, alpha=alpha)
        
        # Run the simulation for the specific (k, n)-combination
        result = simulation_based_performance(
            cov=cov,
            k=k,
            n=n,
            nSim=nSim,
            nMet=nMet,
            dist_function=dist_function,
            rng=rng
        )
        # Store the results in a dict.
        results_dict[(k, n)] = result
    
    # Get columns specifying n and k and concatenate results of individual combinations into a list
    all_rows = []
    # Loop over the dictionary
    for (k,n), df in results_dict.items():
        
        df['n'] = n
        df['k'] = k

        # Append the DataFrame to the list
        all_rows.append(df)
    
    # Concatenate all the rows vertically into one DataFrame
    results_df = pd.concat(all_rows, ignore_index=True)
    
    # Formatting of resulting data frame
    cols = list(df.columns)
    reordered_cols = cols[-2:] + cols[:-2]
    results_df = results_df[reordered_cols]
    results_df = results_df.round(4)
    results_df = results_df.sort_values(by=['n', 'k'], ascending=[True, True])

    return results_df

def plot_n_simulation(nSim_values, y_values, y_label, legend_flag = True):
    """
    This function plots the effect of number of simulation iterations against
    specified performance metrics.

    Parameters
    ----------
    nSim_values : numpy.ndarray
        Vector specifying different numbers of simulations tested.
    y_values : numpy.ndarray
        Data of performance metric for matching simulation number.
    y_label : numpy.ndarray
        Name of performance metric.
    legend_flag : bool
        Specifies whether resulting plot has a legend.
    Returns
    -------
        plot
    """
    cov_est = ['Sample', 'ERCE', 'RERCE', 'LSCE', 'NLSCE']
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(nSim_values, y_values[i], marker='o', linestyle='-', label=cov_est[i])
    plt.axvline(x=100, color='red', linestyle='--')
    plt.xlabel('S (Number of Simulations)', fontsize=16)
    plt.ylabel(y_label, fontsize=16)

    if legend_flag:
        plt.legend(title='Method')
    plt.grid(True)
    plt.show()