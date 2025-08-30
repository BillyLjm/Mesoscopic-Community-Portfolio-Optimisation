# Mesoscopic Community Portfolio Optimisation

*written by [Billy Lim Jun Ming](https://github.com/BillyLjm) and [Manuel Luci](https://github.com/manueluciwqu)*

This was submitted for the WorldQuant University, Masters of Financial Engineering, Capstone project.

## Overview

This repository contains a single `Portfolio` class that implements a pipeline for constructing portfolios using:
1. **Mesoscopic filtering** of correlation matrices (Laloux et al., 1999).
2. **Community detection** to aggregate assets into clusters, via various methods.
3. **Equal weight within clusters** for simplicity
3. **Global Minimum Variance (GMV) optimisation** to optimise the weight between clusters.

## Design rationale

The approach is intended for equity or asset universes where noisy sample correlations obscure true latent structure.

- **Mesoscopic filter** removes the largest eigenvalue (market mode) and eigenvalues within Marcenko-Pastur bounds (interpreted as noise). The retained eigenmodes represent structure across subsets of assets (sectors, factors) and produce a less noisy covariance.
- **Community aggregation** diversifies the idiosyncratic risk within each cluster, and makes the covariance between clusters more stable over time.
- **Equal intra-cluster allocation** is a design choice, and simplifies implementation and reduces estimation error. If desired, intra-cluster optimisation (within-cluster GMV or risk parity) can be added.
- **GMV optimisation** minimises the predicted volatility of the portfolio based on our filtered and ideally more accurate covariance matrix.

## File/class summary

### portfolio.py

`Portfolio` — main class. Important public methods and attributes:
- `__init__(price_data, sectors, algo='Louvain')`
	- Runs mesoscopic filter, community detection (using `algo`), and computes GMV weights based on the given the time-series prices of the assets.
- `mesoscopic_decompose(start=None, end=None)`
	- Returns the eigenspectrum after the mesoscopic filter; listing the eigenvalues, eigenvectors, and component labels (Random Noise / Mesoscopic / Market).
- `mesoscopic_filter()`
	- Applies the mesoscopic filter on the correlation and covariance matrices and stores them to `self.corr` and `self.cov`.
- `cumulative_risk(start=None, end=None)`
	- Returns total variance explained by each filtered component (Random Noise / Mesoscopic / Market).
- `rolling_cumulative_risk(window=252, step=5, n_jobs=-1)`
	- Computes rolling cumulative risks, to determine the evolution of the each filtered component over time.
- `community_detection(algo='Louvain', **kwargs)`
	- Detects communities; sets `self.communities` (dict mapping ticker -> community id) and re-optimises the GMV.
- `gmv_portfolio(short=False)`
	- Solves a convex optimisation for GMV weights at community level, converts community weights back to assets and stores `self.weights`.

Attributes filled during init (and refreshable by calling corresponding methods):

- `price_data` — dataframe of asset prices (aligned with sectors keys
- `returns` — pct_change() of price_data
- `stddev` — per-asset volatility (sample std)
- `corr` — mesoscopic correlation (pd.DataFrame)
- `cov` — mesoscopic covariance (pd.DataFrame)
- `communities` — dict ticker -> community label
- `weights` — dict ticker -> portfolio weight

### mesoscopic_community_portfolio.ipynb

- `mesoscopic_community_portfolio.ipynb`: applies the `portfolio` class to the sp500, and generates the plots for our report
- `mesoscopic_community_portfolio.py`: code version of the ipynb for version logging; generated via jupytext.

## Numerical considerations & caveats

1. **T / N ratio**: The Marčenko–Pastur bound and the stability of sample covariance depend strongly on the ratio T/N (time series length / number of assets). Small T relative to N reduces reliability.
2. **Degenerate clusters**: Some clustering algorithms (DBSCAN) may return noise labels (`-1`). The code currently shifts DBSCAN labels by `+1` to include them; review this choice for your data.
3. **Missing data**: Price data with NaNs will propagate into returns and correlations. The current pipeline uses `pct_change()` and `DataFrame.corr()` default behaviours; pre-cleaning (forward/backfill or pairwise deletion) should be applied depending on your discipline.