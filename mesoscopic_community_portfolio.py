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

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from portfolio import Portfolio
from backtest import fine_tuning_k_means, backtest_fun, plot_results_backtest

# %% [markdown]
# # Define Investment Universe
#
# We are focused on portfolio optimisation, and not stock picking.\
# Thus for simplicty, our investment universe will be the current S&P 500 constituents with daily price data from 2000 to 2024.

# %%
# Get current constituents of S&P 500
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500

# %%
# Get price data from yfinance
prices = yf.download(sp500['Symbol'].tolist(), auto_adjust=False,
                     start='2000-01-01', end='2024-12-31')['Adj Close']
prices.head()

# %%
# Filter only stocks that have daily price data throughout time period
prices = prices.dropna(axis=1)
prices.to_csv('data/prices.csv')

sp500 = sp500[sp500['Symbol'].isin(prices.columns)]
sp500 = sp500.sort_values(by='Symbol').reset_index(drop=True)
sp500.to_csv('data/sp500.csv')

sp500

# %% [markdown]
# # Community Portfolio

# %%
# data_csv = pd.read_csv('data/SandP500.csv')
# data_csv.head()

data_csv = sp500[['Symbol', 'Security', 'GICS Sector']]
data_csv.columns = ['Symbol', 'Name', 'Sector']
data_csv = data_csv.sort_values(by='Symbol').reset_index(drop=True)
data_csv.head()

# %%
# price_data =  pd.read_csv('data/price_data.csv', index_col=0)

price_data =  pd.read_csv('data/prices.csv', index_col=0)
price_data.head()

# %%
portfolio = Portfolio(data_csv, price_data = price_data)
portfolio.compute_return()

# %% [markdown]
# ## Stability

# %%
portfolio.plot_stability()

# %% [markdown]
# ## Community

# %%
for i in ('Louvain', 'Label', 'Kmean'):
    portfolio.mesoscopic_filter()
    portfolio.community_discover(algo = i)
    portfolio.plot_communities_pie(title = i)
    plt.show()

# %% [markdown]
# ## Portfolio Building

# %% [markdown]
# ### Weight Distribution asset

# %%
# create portfolio class
num_asset = 50
price_data_sampled = price_data.sample(n= num_asset, axis=1, random_state=42)
price_data_sampled = price_data_sampled[sorted(price_data_sampled.columns)]
returns = price_data_sampled.pct_change().dropna() 
portfolio = Portfolio(data_csv, price_data = price_data_sampled, returns = returns)
portfolio.C_g = portfolio.mesoscopic_filter()
weights_dict = {}

# Find weights of each portfolio
#alg_list = ["Louvain", "Label", "Kmean", "DBSCAN", "GMV"]
alg_list = ["Louvain", "Label", "Kmean", "DBSCAN"]
for alg in alg_list: 
    communities = portfolio.community_discover(algo=alg)
    weights = portfolio.portfolio_building(community=True, algo=alg)
    weights_dict[alg] = weights
weights_dict['GMV'] = portfolio.portfolio_building(community=False)
weights_df = pd.DataFrame(weights_dict, index=price_data_sampled.columns)

# Plot
labels = weights_df.index.to_list()           
algorithms = weights_df.columns.to_list()     
n_algos = len(algorithms)
x = np.arange(len(labels))                    
width = 0.15                                  

fig, ax = plt.subplots(figsize=(16, 6))
for i, algo in enumerate(algorithms):
    bar_positions = x + (i - n_algos/2) * width + width/2
    ax.bar(bar_positions, weights_df[algo], width=width, label=algo, edgecolor='black')

ax.axhline(1/num_asset, color='gray', linestyle='--', linewidth=1.5, label='1/N')
ax.set_title("Barplot different portfolio weight per algorithm", fontsize=14)
ax.set_ylabel("Weight", fontsize=12)
ax.set_xlabel("Asset", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend(title="Algorithm")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Back-test

# %% [markdown]
# ### Fine-Tuning

# %%
model = ["Louvain", "Label","Kmean","DBSCAN", "Sector","GMV", "Equal"]
results = {}
for m in model:
    print(f"====== Start {m} ======")
    portfolio = Portfolio(data_csv, price_data = price_data)
    results[f"{m}"] = backtest_fun(portfolio, train_months = 36 , burn_period_month  = 6, test_months =18, model = m, short = False, verbose = True, fine_tuning = True)

method_labels = {
        "Louvain": "Mesoscopic community - Louvain",
        "Label": "Mesoscopic community - Label",
        "Kmean": "Mesoscopic community - Kmean",
        "DBSCAN": "Mesoscopic community - DBSCAN",
        "Sector": "Sector community",
        "GMV": "GMV",
        "Equal": "1/N"
    }

markers = {
        "Louvain": "*",
        "Label": "D",
        "Kmean": "s",
        "DBSCAN": "<",
        "Sector": "^",
        "GMV": "p",
        "Equal": "X"
    }

colors = {
        "Louvain": "red",
        "Label": "purple",
        "Kmean": "black",
        "DBSCAN": "orange",
        "Sector": "grey",
        "GMV": "blue",
        "Equal": "green"
    }

summary_df = plot_results_backtest(results, method_labels, markers, colors)

# %%
model = ["Louvain", "Label", "Kmean", "DBSCAN", "Sector","GMV", "Equal"]
results_short = {}
for m in model:
    print(f"====== Start {m} ======")
    portfolio = Portfolio(data_csv, price_data = price_data)
    results_short[f"{m}"] = backtest_fun(portfolio, train_months = 36 , burn_period_month  = 6, test_months =18, model = m, short = True, verbose = True, fine_tuning = True)
method_labels = {
        "Louvain": "Mesoscopic community - Louvain",
        "Label": "Mesoscopic community - Label",
        "Kmean": "Mesoscopic community - Kmean",
        "DBSCAN": "Mesoscopic community - DBSCAN",
        "Sector": "Sector community",
        "GMV": "GMV",
        "Equal": "1/N"
    }

markers = {
        "Louvain": "*",
        "Label": "D",
        "Kmean": "s",
        "DBSCAN": "<",
        "Sector": "^",
        "GMV": "p",
        "Equal": "X"
    }

colors = {
        "Louvain": "red",
        "Label": "purple",
        "Kmean": "black",
        "DBSCAN": "orange",
        "Sector": "grey",
        "GMV": "blue",
        "Equal": "green"
    }

summary_df_short = plot_results_backtest(results_short, method_labels, markers, colors)
