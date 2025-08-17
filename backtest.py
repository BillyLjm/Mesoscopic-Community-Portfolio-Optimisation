import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from portfolio import Portfolio

def fine_tuning_k_means(data_csv, portfolio_train,split = 0.8, list_cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], short = False, verbose = True):
    train_size = int(len(portfolio_train.returns) * split)
    train_df = portfolio_train.returns.iloc[:train_size]
    val_df = portfolio_train.returns.iloc[train_size:].values
    sharp_ratio = []
    for cluster in list_cluster:
        portfolio_train = Portfolio(data_csv, price_data = portfolio_train.price_data, returns = train_df)
        weights = portfolio_train.portfolio_building(short = short, community = True, algo = 'Kmean', n_clusters = cluster)
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

def backtest_fun(portfolio, train_months = 36 , burn_period_month  = 6, test_months =18, model = 'Louvain', short = False, verbose = True, **kwargs):
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