# -*- coding: utf-8 -*-
"""
@author: Dario Tassone
"""

import pandas as pd
import yfinance as yf

# Fetch the current list of S&P 500 companies from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
smp500_table = pd.read_html(url, header=0)[0]

# Extract tickers
tickers = smp500_table['Symbol'].tolist()
tnx = yf.Ticker("^TNX")

# Specify start and end date
start_date = '1993-12-30'
end_date = '2024-01-31'

# Download data using yfinance
data = yf.download(tickers, start=start_date, end=end_date, interval='1mo')['Adj Close']
tnx_data = tnx.history(start=start_date, end=end_date, interval="1mo")

# Filter out assets without data before the start date
columns_with_na_in_first_row = data.iloc[0].isna()
data_filtered = data.loc[:, ~columns_with_na_in_first_row]

# Sample 200 assets of the remaining assets
data_random = data_filtered.sample(n=200, axis=1, random_state=42)
nan_count = data_random.isna().sum()

# Fill empty cells with the prices from the previous period
data_random_filled = data_random.fillna(method='ffill')
nan_count_after = data_random.isna().sum()

# Compute returns as percentage changes
data_return= data_random_filled.pct_change()
nan_count_return = data_return.isna().sum()
data_return = data_return.drop(index=data_return.index[0])
tnx_data = tnx_data['Close']/100

# Reconcile date time
data_return.index = pd.to_datetime(data_return.index).tz_localize(None)
tnx_data.index = pd.to_datetime(tnx_data.index).tz_localize(None)

# Convert tnx_data (a Series) to a DataFrame
tnx_data_df = tnx_data.to_frame()
tnx_data_df.columns = ['10yr_Treasury_Yield']

# Merge return data of assets and risk-free gov't bond
merged_data = pd.merge(data_return, tnx_data_df, how='inner', left_index=True, right_index=True)

merged_data.to_csv(r'C:\Users\dario\OneDrive\Dokumente\MBF\Master Thesis\Analysis\return_data.csv', index=True)