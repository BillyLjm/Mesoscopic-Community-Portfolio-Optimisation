import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from portfolio import Portfolio

def fine_tuning_k_means(data_csv, portfolio_train,split = 0.8, list_cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], short = False, verbose = True):
    """
    Find the optimal number of KMeans clusters (communities) for portfolio
    construction by maximizing out-of-sample Sharpe Ratio.

    For each cluster number in `list_cluster`, the function:
      - Fits a portfolio using KMeans-based clustering on a training dataset
      - Tests the resulting portfolio on the validation (out-of-sample) split,
      - Computes and collects the annualized Sharpe ratios,
      - Optionally prints an intermediate summary table for each cluster number.

    The function returns the optimal number of clusters (from `list_cluster`)
    corresponding to the highest Sharpe ratio in validation.

    Parameters
    ----------
    data_csv : pandas.DataFrame
        Dataframe containing metadata (such as 'Symbol' and 'Sector') for all assets.
    portfolio_train : Portfolio
        A Portfolio object already containing return data.
    split : float, optional
        Fraction of samples to use for the training split (default is 0.8).
    list_cluster : list of int, optional
        List of KMeans cluster numbers to test (default is [2, 3, ..., 12]).
    short : bool, optional
        If True, allows short selling in the portfolio optimization (default is False).
    verbose : bool, optional
        If True, prints a summary DataFrame for every cluster number (default is True).

    Returns
    -------
    best_n_clusters : int
        The number of clusters in `list_cluster` which yields the highest
        annualized Sharpe ratio on the validation (out-of-sample) set.

    Notes
    -----
    Prints summary tables for each KMeans cluster size if `verbose=True`.
    """
    train_size = int(len(portfolio_train.returns) * split)
    train_df = portfolio_train.returns.iloc[:train_size]
    val_df = portfolio_train.returns.iloc[train_size:].values
    sharp_ratio = []
    for cluster in list_cluster:
        portfolio_train = Portfolio(data_csv,
            price_data = portfolio_train.price_data, returns = train_df)
        weights = portfolio_train.portfolio_building(short = short,
            community = True, algo = 'Kmean', n_clusters = cluster)
        test_returns = val_df @ weights
        sharpe_ann = test_returns.mean() / test_returns.std() * np.sqrt(252)
        sharp_ratio.append(sharpe_ann)
        if verbose:
            summary_df = pd.DataFrame([
            {
                "Number Cluster": cluster,
                "Sharp Ratio": sharpe_ann,
            }
        ])
        print(tabulate(summary_df.round(4), headers='keys', tablefmt='fancy_grid'))
    print(f"====== Number Cluster: {list_cluster[np.argmax(sharp_ratio)]}======")
    return list_cluster[np.argmax(sharp_ratio)]

def backtest_fun(portfolio, train_months = 36 , burn_period_month  = 6,
    test_months =18, model = 'Louvain', short = False, verbose = True, **kwargs):
    """
    Perform a rolling-window backtest of various portfolio clustering models.

    For each rolling window:
      - Split data into a training period, burn-in period, and test period.
      - Fit a `Portfolio` on the training period using the specified model.
      - Apply the resulting portfolio weights to the test period.
      - Compute annualized Sharpe ratio and reliability of out-of-sample performance.
      - Optionally prints summary and cluster counts.

    Supported models: 'Louvain', 'Label', 'Equal', 'GMV', 'Kmean', 'DBSCAN', 'Sector'.
    For 'Kmean', may apply optional fine-tuning by maximizing out-of-sample Sharpe ratio.

    Parameters
    ----------
    portfolio : Portfolio
        An instantiated Portfolio object containing the entire return series.
    train_months : int, optional
        Number of months in each rolling window used for training (default is 36).
    burn_period_month : int, optional
        Number of months between training and test (burn-in, default is 6).
    test_months : int, optional
        Number of months to use for out-of-sample test after each training window (default is 18).
    model : {'Louvain', 'Label', 'Equal', 'GMV', 'Kmean', 'DBSCAN', 'Sector'}, optional
        The portfolio clustering/model algorithm to use (default is 'Louvain').
    short : bool, optional
        If True, allow short selling in the optimizer (default is False).
    verbose : bool, optional
        If True, print detailed results and diagnostics for each window (default is True).
    **kwargs
        Optional arguments, including:
            fine_tuning : bool
                If True (for 'Kmean' model), perform fine-tuning of the number
                of clusters via out-of-sample Sharpe ratio.
            list_cluster : list of int
                Used for fine-tuning in KMeans.

    Returns
    -------
    results : list of dict
        Each entry is a dictionary containing metrics for a single out-of-sample period:
          - "start": Timestamp for train window end
          - "end": Timestamp for test window end
          - "Reliability": Relative difference between expected and realized annualized volatility in test
          - "Sharpe Ratio": Annualized Sharpe ratio on test set
          - "Number of Cluster": Number of clusters used/detected (if applicable), else None

    Raises
    ------
    ValueError
        If `model` is not a recognized algorithm string.
    """
    returns = portfolio.compute_return()
    returns.index = pd.to_datetime(returns.index)
    results = []
    start_idx = returns.index[0]
    current_date = start_idx
    while current_date + pd.DateOffset(months=train_months + burn_period_month + test_months) < returns.index[-1]:
        train_end = current_date + pd.DateOffset(months=train_months)
        test_start = train_end + pd.DateOffset(months= test_months + burn_period_month)
        test_end = test_start + pd.DateOffset(months=test_months)
        returns_train = returns[(returns.index >= current_date) & (returns.index < train_end)]
        test_data = returns[(returns.index >= test_start) & (returns.index < test_end)]
        portfolio_train = Portfolio(portfolio.data_csv, price_data = portfolio.price_data, returns = returns_train)
        if model == 'Louvain':
            weights = portfolio_train.portfolio_building(short = short, algo = 'Louvain')
            print(f"Cluster found: {len(np.unique(weights))}")
            num_cluster = len(np.unique(weights))
        elif model == 'Label':
            weights = portfolio_train.portfolio_building(short = short, algo = 'Label')
            print(f"Cluster found: {len(np.unique(weights))}")
            num_cluster = len(np.unique(weights))
        elif model == 'Equal':
            weights = np.ones(len(test_data.columns)) * 1 / len(test_data.columns)
            num_cluster = None
        elif model == "GMV":
            weights = portfolio_train.portfolio_building(short = short, community = False)
            num_cluster = None
        elif model == "Kmean":
            fine_tuning = kwargs.get('fine_tuning', False)
            if fine_tuning is True:
                list_cluster = kwargs.get('list_cluster', [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
                num_cluster = fine_tuning_k_means(portfolio.data_csv, portfolio_train = portfolio_train,split = 0.8, list_cluster = list_cluster , short = short)
                weights = portfolio_train.portfolio_building(short = short, community = True, algo = 'Kmean', n_cluster = num_cluster)
                print(f"Cluster found: {num_cluster}")
            else:
                weights = portfolio_train.portfolio_building(short = short, community = True, algo = 'Kmean')
                print(f"Cluster found: {len(np.unique(weights))}")
                num_cluster = len(np.unique(weights))
        elif model == "DBSCAN":
            weights = portfolio_train.portfolio_building(short = short, community = True, algo = 'DBSCAN')
            print(f"Cluster found: {len(np.unique(weights))}")
            num_cluster = len(np.unique(weights))
        elif model == "Sector":
            weights = portfolio_train.portfolio_building(short = short, community = True, algo = 'Sector')
            print(f"Cluster found: {len(np.unique(weights))}")
            num_cluster = len(np.unique(weights))
        else:
            raise ValueError("Select a model between (Louvain, Label, Equal, GMV, Kmean, DBASCAN, Sector)")
        test_returns = test_data @ weights
        sigma_p = (portfolio_train.returns@ weights).std()
        sharpe_ann = test_returns.mean() / test_returns.std() * np.sqrt(252)
        sigma_p_ann = sigma_p * np.sqrt(252)
        sigma_test_ann = test_returns.std() * np.sqrt(252)
        results.append({
            "start": train_end,
            "end": test_end,
            "Reliability": np.abs(sigma_test_ann - sigma_p_ann)/sigma_p_ann,
            "Sharpe Ratio": sharpe_ann,
            "Number of Cluster": num_cluster
            })
        if verbose:
            print(results[-1])
        current_date = current_date + pd.DateOffset(months = train_months)
    return results

def plot_results_backtest(results, method_labels, markers,colors):
    """
    Plot and compare rolling backtest metrics for different portfolio
    construction methods, and return a summary DataFrame of average results.

    For each model in the results, the function:
      - Converts the results into a DataFrame.
      - Plots the time series of Sharpe Ratio, Reliability, and Number of
        Cluster for each model.
      - Shows a comparison plot and legend for each metric.
      - Prints a summary table with average metrics and returns it as a DataFrame.

    Parameters
    ----------
    results : dict of list of dict
        A dictionary mapping method names to a list of backtest performance
        summaries. Each list entry is a dictionary with at least the keys:
            - 'end'
            - 'Sharpe Ratio'
            - 'Reliability'
            - 'Number of Cluster'
    method_labels : dict
        Mapping from model names (keys in `results`) to human-readable labels
        for plots and tables.
    markers : dict
        Mapping from model names to marker styles for matplotlib plot.
    colors : dict
        Mapping from model names to colors used in matplotlib plot.

    Returns
    -------
    summary_df : pandas.DataFrame
        DataFrame with a row for each model and columns:
            - 'Method': Human-readable model label
            - 'Average Sharpe Ratio': Mean Sharpe ratio across all rolling periods
            - 'Average Reliability': Mean relative volatility difference
            - 'Average Number of Cluster': Mean number of detected clusters

    Notes
    -----
    This function also produces and displays matplotlib figures, and prints
    the summary DataFrame as a formatted table.
    """
    dfs = {model: pd.DataFrame(data) for model, data in results.items()}

    for metric in ['Sharpe Ratio', 'Reliability', 'Number of Cluster']:
        fig, ax = plt.subplots(figsize=(16, 8))
        for model, df in dfs.items():
            ax.plot(
                df['end'], df[metric],
                marker=markers[model],
                linestyle='-.',
                color=colors[model],
                label=method_labels[model]
            )

        ax.set_xticks(dfs["Louvain"]['end'])
        ax.set_xticklabels(dfs["Louvain"]['end'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
        ax.set_title(f"{metric} Comparison Test")
        ax.set_xlabel("End Test Date", size = 12)
        ax.set_ylabel(f"Annualized - {metric}", size = 12)
        ax.grid()
        ax.legend(title = 'Algorithm')
        plt.tight_layout()
        plt.show()

    # Summary Table
    summary_df = pd.DataFrame([
        {
            "Method": method_labels[model],
            "Average Sharpe Ratio ": dfs[model]['Sharpe Ratio'].mean(),
            "Average Reliability": dfs[model]['Reliability'].mean(),
            "Average Number of Cluster": dfs[model]['Number of Cluster'].mean()
        }
        for model in dfs
    ])

    print(tabulate(summary_df.round(4), headers='keys', tablefmt='fancy_grid'))
    return summary_df