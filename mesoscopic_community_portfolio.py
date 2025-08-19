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
# # Investment Universe
#
# We are focused on portfolio optimisation, and not stock picking.\
# Thus for simplicty, our investment universe will be the current S&P 500 constituents with daily price data from 2000 to 2024.

# %%
# %load_ext autoreload
# %autoreload 2
    
import numpy as np
import pandas as pd
import yfinance as yf

import seaborn as sns
import matplotlib.pyplot as plt

from portfolio import Portfolio
from backtest import fine_tuning_k_means, backtest_fun, plot_results_backtest

dir_data = 'data/'
dir_fig = 'fig/'

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
sp500.to_csv('data/sp500.csv')

sp500

# %% [markdown]
# # Mesoscopic Community Portfolio

# %% [markdown]
# ## Create Porfolio

# %%
# sectors = pd.read_csv(dir_data + 'SandP500.csv')
# sectors.head()

sectors = pd.read_csv(dir_data + 'sp500.csv')
sectors = sectors[['Symbol', 'Security', 'GICS Sector']]
sectors.columns = ['Symbol', 'Name', 'Sector']
sectors = sectors.sort_values(by='Symbol').reset_index(drop=True)
sectors.head()

# %%
# price_data =  pd.read_csv(dir_data + 'price_data.csv', index_col=0)

price_data =  pd.read_csv('data/prices.csv', index_col=0)
price_data.head()

# %%
portfolio = Portfolio(price_data, dict(zip(sectors['Symbol'], sectors['Sector'])))

# %% [markdown]
# ## Mesocopic Correlation

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

# %% [markdown]
# ## Community

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

# %%
# portfolio = Portfolio(price_data, dict(zip(sectors['Symbol'], sectors['Sector'])))
for algo in ('Louvain', 'Label Propagation', 'Agglomerative', 'DBSCAN', 'Kmean'):
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
    communities = portfolio.community_detection(algo=alg)
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
