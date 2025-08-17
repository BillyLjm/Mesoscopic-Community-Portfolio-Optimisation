import numpy as np
import pandas as pd
import networkx as nx
import cvxpy as cp
from collections import Counter, defaultdict

import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances

class Portfolio:
    def __init__(self, data_csv, start="2000-01-01", end="2024-12-31", price_data=None, returns = None):
        self.data_csv = data_csv
        self.symbols = self.data_csv['Symbol']
        self.sector = self.data_csv['Sector']
        self.start = start
        self.end = end
        self.price_data = price_data
        self.returns = returns
        self.dates = None
        self.C_g = None
        self.D = None
        self.cov_g = None
        self.communities = None
        self.weight = None

    def get_data_yahoo(self, save = False, path = None, no_nan = True):
        self.price_data = yf.download(self.symbols.to_list(), self.start , self.end, auto_adjust=False)['Adj Close']
        if no_nan:
            self.price_data.dropna(axis=1, inplace = True)
        if save:
            if path is None:
                raise ValueError("Path must be specified if save is True.")
            self.price_data.to_csv(path)


    def _evaluate_cov_g(self):
        self.D = np.diag(self.returns.std().values)
        self.cov_g = self.D @ self.C_g @ self.D
        return self.cov_g

    def compute_return(self):
        if self.price_data is None and self.returns is None:
             raise ValueError("Price data not available, download or use a valid dataframe")
        elif self.returns is None:
            self.price_data.dropna(axis = 1, inplace = True)
            self.returns = self.price_data.pct_change().dropna()
        self.symbols =  self.returns.columns.to_list()
        self.dates = pd.to_datetime(self.returns .index)
        self.sector = self.data_csv[self.data_csv['Symbol'].isin(self.returns .columns)]['Sector'].to_list()
        return self.returns

    def mesoscopic_filter(self):
        if self.returns  is None:
            raise ValueError("No return, please before compute return")
        corr_matrix = self.returns.corr().values
        T, N = self.returns .shape
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        Q = T / N
        lambda_max = (1 + 1 / np.sqrt(Q))**2
        lambda_1 = eigvals[-1] #isolate the biggest eigenvalue
        sigma2 = 1 - lambda_1 / N
        lambda_max = sigma2 * (1 + 1 / np.sqrt(Q))**2 #Laloux correction
        meso_indices = np.where((eigvals > lambda_max) & (eigvals < lambda_1))[0]
        C_g = np.zeros((N, N))
        for i in meso_indices:
            C_g += eigvals[i] * np.outer(eigvecs[:, i], eigvecs[:, i])
        self.C_g = C_g
        self.cov_g = self._evaluate_cov_g()
        return self.C_g



    @staticmethod
    def _get_risk_fractions(returns_window):
        T, N = returns_window.shape
        corr = returns_window.corr().values

        eigvals, _ = np.linalg.eigh(corr)
        eigvals = np.sort(eigvals)

        Q = T / N
        lambda_1 = eigvals[-1]
        sigma2 = 1 - lambda_1 / N
        lambda_max = sigma2 * (1 + 1 / np.sqrt(Q))**2

        noise = eigvals[eigvals <= lambda_max]
        market = np.array([lambda_1])
        meso = eigvals[(eigvals > lambda_max) & (eigvals < lambda_1)]

        total_risk = np.sum(eigvals)

        return {
            "noise": np.sum(noise) / total_risk,
            "market": np.sum(market) / total_risk,
            "meso": np.sum(meso) / total_risk,
        }

    def plot_stability(self, windows_years = 2):

        if self.dates is None:
            raise ValueError("No date available, please run compute_return first")
        windows = []
        start = self.dates[0]

        while start + pd.DateOffset(years = windows_years) < self.dates[-1]:
            end = start + pd.DateOffset(years = windows_years)
            windows.append((start, end))
            start = end
        risks = []

        for start, end in windows:
            returns = np.exp(self.returns ) - 1
            returns.index = pd.to_datetime(returns.index)
            subset = returns[(returns.index >= start) & (returns.index < end)]
            risks.append(self._get_risk_fractions(subset))

        risk_df = pd.DataFrame(risks)
        risk_df['window_start'] = [w[0] for w in windows]
        risk_df.set_index('window_start', inplace=True)
        diffs = risk_df.diff().dropna()
        std_noise = diffs['noise'].std()
        std_market = diffs['market'].std()
        std_meso = diffs['meso'].std()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(risk_df.index, risk_df['noise'], label='Noise', color = "red", marker = 'o', linestyle = 'dashed')
        ax.plot(risk_df.index, risk_df['market'], label='Systemic', color = "blue", marker = 'o', linestyle = 'dashed')
        ax.plot(risk_df.index, risk_df['meso'], label='Mesoscopic', color = "green", marker = 'o', linestyle = 'dashed')


        ax.text(risk_df.index[3], risk_df['noise'].iloc[3], rf'$\sigma_r={std_noise:.3f}$', va='bottom', ha='left', size = 12,bbox=dict(facecolor='white', alpha=0.5))
        ax.text(risk_df.index[3], risk_df['market'].iloc[3], rf'$\sigma_s={std_market:.3f}$', va='bottom', ha='right', size = 12,bbox=dict(facecolor='white', alpha=0.5))
        ax.text(risk_df.index[3], risk_df['meso'].iloc[3], rf'$\sigma_g={std_meso:.3f}$', va='bottom', ha='left', size = 12,bbox=dict(facecolor='white', alpha=0.5))

        ax.set_title('Cumulative Risk Fractions over Time')
        ax.set_ylabel('Fraction of Total Risk')
        ax.set_xlabel('Start of 2-Year Window')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    @staticmethod
    def _commu_to_list(G, communities):
        node_to_comm = {}
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                node_to_comm[node] = comm_id
        n_assets = len(G.nodes)
        communities = [node_to_comm[i] for i in range(n_assets)]
        return communities

    @staticmethod
    def kmeans_C_g(C_g, n_clusters):
        labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(C_g)
        return labels

    @staticmethod
    def DBSCAN_Cg(C_g, returns, eps=0.5, min_samples=5, metric='precomputed'):
        X = (returns @ C_g).T
        dist = pairwise_distances(X, metric='correlation')
        labels = DBSCAN(eps=0.3, min_samples=3, metric='precomputed').fit_predict(dist)
        return labels

    @staticmethod
    def sector_to_label(list_sector):
        # Crea la mappatura settore → numero
        sector_to_number = {sector: i for i, sector in enumerate(set(list_sector))}

        # Applica la mappatura
        mapped_sectors = [sector_to_number[sector] for sector in list_sector]

        return mapped_sectors

    def community_discover(self, algo = "Louvain", seed = 42, **kwargs):
        if self.C_g is None:
            raise ValueError("No Mesoscopic structure found, please run mesoscopic_filter")
        G = nx.from_numpy_array(self.C_g)
        if algo == "Louvain":
            communities = nx.community.louvain_communities(G, seed=seed, weight='weight')
            self.communities = self._commu_to_list(G, communities)
        elif algo == "Label":
                    communities = list(nx.community.fast_label_propagation_communities(G, weight='weight',seed = seed))
                    self.communities = self._commu_to_list(G, communities)
        elif algo == "Kmean":
                    n_clusters = kwargs.get("n_clusters", 5)
                    self.communities = self.kmeans_C_g(self.C_g, n_clusters)
        elif algo == "DBSCAN":
                    min_samples = kwargs.get("min_samples", 20)
                    eps = kwargs.get("eps", 0.5)
                    metric = kwargs.get("metric", "precomputed")
                    self.communities = self.DBSCAN_Cg(returns = self.returns, C_g = self.C_g, eps = eps, min_samples = min_samples, metric = metric)
        elif algo == "Sector":
             self.communities = self.sector_to_label(self.sector)
        else:
            raise ValueError("Select correct algorithm for community discover")


        return self.communities

    def plot_communities_pie(self, title = 'Louvain'):
        if self.communities is None:
            raise ValueError("No communities found, please run community_discover")
        comm2sectors = defaultdict(list)
        for asset_idx, comm_id in enumerate(self.communities):
            comm2sectors[comm_id].append(self.sector[asset_idx])

        sector_labels = sorted(set(self.sector))
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        sector_color_dict = dict(zip(sector_labels, colors))

        n_communities = len(comm2sectors)
        n_cols = min(n_communities, 3)
        n_rows = (n_communities + n_cols - 1) // n_cols

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axs = axs.flatten() if n_communities > 1 else [axs]
        num_com = 1
        for i, (comm_id, sectors) in enumerate(comm2sectors.items()):

            ax = axs[i]
            count = Counter(sectors)
            sizes = [count.get(s, 0) for s in sector_labels]
            total = sum(sizes)
            sizes = [s / total for s in sizes] if total > 0 else [0] * len(sizes)

            # Explode: separa le fette grandi per leggibilità
            explode = [0.05 if s > 0.15 else 0 for s in sizes]

            wedges, texts, autotexts = ax.pie(
                sizes,
                colors=[sector_color_dict[s] for s in sector_labels],
                startangle=90,
                wedgeprops=dict(edgecolor='white'),
                autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                pctdistance=0.8,
                labeldistance=1.1,
                explode=explode
            )

            for text in autotexts:
                text.set_fontsize(12)
                text.set_horizontalalignment('center')
                text.set_verticalalignment('center')
                text.set_weight('bold')

            ax.set_title(f"Community {num_com}", fontsize=14)
            num_com += 1
            ax.axis('equal')

        fig.suptitle(f"Community Searching Algorithm {title}", fontsize = 14)


        for j in range(i + 1, len(axs)):
            axs[j].axis('off')


        fig.legend(
            handles=[plt.Line2D([0], [0], marker='o', color='w', label=lab,
                                markerfacecolor=sector_color_dict[lab], markersize=10)
                    for lab in sector_labels],
            loc='lower center',
            ncol=min(len(sector_labels), 5),
            bbox_to_anchor=(0.5, -0.05),
            title = 'Sector'
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def solve_gmv(cov_g, short = False):
        N = cov_g.shape[0]
        w = cp.Variable(N)
        objective = cp.Minimize(cp.quad_form(w, cov_g))
        if short:
            constraints = [cp.sum(w) == 1]
        else:
            constraints = [cp.sum(w) == 1,  w >= 0 ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return w.value

    @staticmethod
    def solve_gmv_community(cov_g, community_labels, short=False):
        N = len(community_labels)

        # Ricostruisci: community_id → [asset indices]
        community_map = defaultdict(list)
        for idx, c_id in enumerate(community_labels):
            community_map[c_id].append(idx)

        community_list = list(community_map.values())
        n = len(community_list)

        Sigma_c = np.zeros((n, n))

        for c in range(n):
            assets_c = community_list[c]
            Nc = len(assets_c)

            var_c = np.mean([cov_g[i, i] for i in assets_c])
            cov_c = np.mean([cov_g[i, j] for i in assets_c for j in assets_c if i != j]) if Nc > 1 else 0
            Sigma_c[c, c] = var_c + (Nc - 1) * cov_c

            for k in range(c + 1, n):
                assets_k = community_list[k]
                cross_cov = np.mean([cov_g[i, j] for i in assets_c for j in assets_k])
                Sigma_c[c, k] = Sigma_c[k, c] = cross_cov

        W = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(W, Sigma_c))
        if short:
            constraints = [cp.sum(W) == 1]
        else:
            constraints = [cp.sum(W) == 1,  W >= 0 ]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = np.zeros(N)
        for c, Wc in enumerate(W.value):
            assets_c = community_list[c]
            Nc = len(assets_c)
            for i in assets_c:
                weights[i] = Wc / Nc

        return weights

    def portfolio_building(self, seed = 42,**kwargs ):
        use_community = kwargs.get("community", True)
        is_equal = kwargs.get("is_equal", False)
        if is_equal and use_community:
             raise ValueError("use_community and is_equal cannot be True at same time ")
        short = kwargs.get("short", False)
        algo = kwargs.get("algo", "Louvain")
        seed = kwargs.get("seed", 42)
        min_samples = kwargs.get("min_samples", 5)
        eps = kwargs.get("eps", 0.5)
        metric = kwargs.get("metric", "precomputed")
        self.compute_return()
        if use_community:
            self.mesoscopic_filter()
            algo = kwargs.pop("algo", "Louvain")
            self.communities = self.community_discover(algo = algo, seed = seed, **kwargs)
            self.w = self.solve_gmv_community(self.cov_g,self.communities, short = short)
        elif is_equal:
             self.w = np.ones(len(self.returns.columns)) * 1 / len(self.returns.columns)
        else:
            self.cov_g = self.returns.cov().values
            self.w = self.solve_gmv(self.cov_g, short)
        return self.w