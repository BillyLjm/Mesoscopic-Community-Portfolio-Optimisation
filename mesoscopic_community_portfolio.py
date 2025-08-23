# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mescoscopic Community Portfolio

# %%
import numpy as np
import pandas as pd
import yfinance as yf

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict

from portfolio import Portfolio

dir_data = 'data/'
dir_fig = 'fig/'

# %% [markdown]
# ## Investment Universe
#
# We are focused on portfolio optimisation, and not stock picking.\
# Thus for simplicty, our investment universe will be the current S&P 500 constituents with daily price data from 2000 to 2024.
#
# The code below downloads the relevant data from the internet.

# %%
# Get current constituents of S&P 500
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

# Get price data from yfinance
prices = yf.download(sp500['Symbol'].tolist(), auto_adjust=False,
                     start='2000-01-01', end='2024-12-31')['Adj Close']

# Filter only stocks that have daily price data throughout time period
prices = prices.dropna(axis=1)
prices.to_csv('data/prices.csv')

sp500 = sp500[sp500['Symbol'].isin(prices.columns)]
sp500 = sp500.sort_values(by='Symbol').reset_index(drop=True)
sp500.to_csv('data/sp500.csv', index=False)

sp500

# %% [markdown]
# ## Create Porfolio
#
# With the data downloaded, we create the Portoflio class which does all the claculations and optimisations.

# %%
# create sector mapping
sectors = pd.read_csv(dir_data + 'sp500.csv')
sectors = dict(zip(sectors['Symbol'], sectors['GICS Sector']))

# read price data
price_data =  pd.read_csv(dir_data + 'prices.csv', index_col=0)
price_data.index = pd.to_datetime(price_data.index)

# create portfolio class
portfolio = Portfolio(price_data, sectors)

# %% [markdown]
# ## Mesocopic Correlation
#
# We applied the Laloux corrections to the correlation matrix to yield the mesoscopic correlation matrix

# %%
# plot eigenspectrum deomposition
eigvals, _, components, lambda_max, lambda_1 = portfolio.mesoscopic_decompose()
df = pd.DataFrame({'component': components, 'eigval': eigvals})
df = df.pivot(columns='component', values='eigval')

fig, axs = plt.subplots(1, 2, figsize=(10,5), sharey=True)
# left tail
df[df.sum(axis=1) < lambda_max+1].plot.hist(
    bins=50, density=False, stacked=True, legend=False, ax=axs[0])
axs[0].set_title('Left Tail')
# right tail
df.iloc[-10:].plot.hist(bins=50, density=False, stacked=True, ax=axs[1])
axs[1].legend(title='Component')
axs[1].set_title('Right Tail')
# overall
plt.suptitle('Eigenspectrum of Correlation Matrix')
plt.tight_layout()
plt.savefig(dir_fig + 'meso_corr_spectrum.png')
plt.show()

# %%
# calculate cumulative risks
risks = portfolio.rolling_cumulative_risk(window=252, step=1)
risks_frac = risks.div(risks.sum(axis=1), axis=0)

# plot
risks_frac.plot(figsize=(12,6))
plt.title('Rolling 1-year Cumulative Risk\nFraction by Component (Daily Returns)')
plt.xlabel('Window Start Date')
plt.ylabel('Risk Fraction\n(Daily Returns)')
plt.legend(title='Component')
plt.savefig(dir_fig + 'meso_corr_rolling.png')
plt.show()

# %% [markdown]
# ## Community Detection
#
# Applying the various algorithms to cluster the stocks together based on the mescoscopic correlation matrix

# %%
# find communities & plot GICS sector composition
for algo in ('Louvain', 'Label Propagation', 'Agglomerative', 'DBSCAN'):
    # discover community
    portfolio.community_detection(algo)
    df = pd.DataFrame({'community': portfolio.communities, 'sector': portfolio.sectors})
    df = pd.pivot_table(df, index='sector', columns='community', aggfunc='size').fillna(0)
    df.columns = df.columns.astype(int)
    
    # create subplots
    ncols = 3
    nrows = int(np.ceil(df.shape[1] / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, nrows*5))
    axs = axs.flatten()
    for j in range(df.shape[1] , len(axs)): fig.delaxes(axs[j])
    
    # plot pies charts
    for i, col in enumerate(df.columns):
        wedges, texts, autotexts = axs[i].pie(
            df[col], labels=None, startangle=90,
            autopct=lambda x: f'{x:.0f}%' if x >= 10 else '',
            colors=plt.cm.tab10.colors + ((1, 0.87, 0.13),),
            wedgeprops=dict(edgecolor='white', linewidth=1.5),
            textprops=dict(color='white', fontsize=12, fontweight='bold'),
        )
        axs[i].set_title(f'Community {col}')

    # figure clean-up
    if algo in ('Louvain', 'Label Propagation'):
        algo_full = algo + ' Community Detection'
    else:
        algo_full = algo + ' Clustering'
    fig.suptitle(f'Communities Identifed by \n{algo_full}\n')
    fig.legend(df.index, title='GICS Sector', loc='lower center', ncol=4)
    plt.tight_layout()
    plt.savefig(dir_fig + f'communities_{algo}.png')
    plt.show()

# %% [markdown]
# ## Back-Testing
#
# We optimise the weights between communities to yield the global-minimum-variance (GMV) portfolios, and keep equal weights within portfolios.  
# These portfolios are then tested using an anchored walk-forward method, with a 1-year testing period, and a 3-year burn-in period

# %%
ret_data = price_data.pct_change()

# determine test start dates
period_test = pd.DateOffset(years=1)
period_burn = pd.DateOffset(years=3)
test_dates = pd.date_range(price_data.index.min() + period_burn, 
                           price_data.index.max() - period_test,
                           freq=pd.DateOffset(years=1))

# anchored walk-forward
returns = defaultdict(list)
reliability = defaultdict(list)
ncomm = defaultdict(list)
for algo in ('Louvain', 'Label Propagation', 'Agglomerative', 'DBSCAN', 'Sector', 'Equal-Weight', 'None'):
    pbar = tqdm(test_dates)
    pbar.set_description(algo)
    for test_date in pbar:
        # train portfolio optimisation
        portfolio = Portfolio(price_data[:test_date], sectors, algo=algo)
        # calculate returns in test period
        ret_test = ret_data[test_date:test_date + period_test]
        wts = ret_test.columns.map(portfolio.weights)
        ret_test = ret_test @ wts
        returns[algo].append(ret_test)
        # calculate reliability (via stdev in train dataset)
        ret_train = ret_data[:test_date ]
        wts = ret_train.columns.map(portfolio.weights)
        ret_train = ret_train @ wts
        reliability[algo].append(np.abs(np.std(ret_test)/np.std(ret_train) - 1))
        # remember the number of communities
        ncomm[algo].append(len(set(portfolio.communities.values())))
    returns[algo] = pd.concat(returns[algo]) # concat pandas time-series

returns = pd.DataFrame(returns)
reliability = pd.DataFrame(reliability)
ncomm = pd.DataFrame(ncomm)
returns.head()

# %%
# calculate aggregate metrics
summary = {}
summary['Sharpe Ratio'] = returns.mean() / returns.std() * np.sqrt(252)
summary['Mean Reliability'] = reliability.mean()
summary['Median Reliability'] = reliability.median()
summary['Mean Num Communities'] = ncomm.mean()
summary['Median Num Communities'] = ncomm.median()
summary = pd.DataFrame(summary)
summary.to_csv(dir_fig + 'sumamry.csv')
summary

# %%
# plot returns
cumlogret = np.log((1 + returns).cumprod()) 
cumlogret.plot()
plt.axhline(0, color='k')
plt.title('Cumulative Log Returns')
plt.ylabel('Log Returns')
plt.savefig(dir_fig + 'cum_returns.png')
plt.show()
