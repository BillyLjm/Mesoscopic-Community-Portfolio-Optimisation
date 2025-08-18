import numpy as np
import pandas as pd
import networkx as nx
import cvxpy as cp

import warnings
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances

class Portfolio:
    """
    Portfolio construction using mesoscopic filtering and community detection.

    Parameters
    ----------
    price_data : pd.DataFrame
        Price data with tickers as columns and timesteps as rows.
    sectors : dictionary
        Maps from ticker to sector.
    """
    def __init__(self, price_data, sectors):
        # filter inner join of price_data and sectors
        tmp = set(price_data.columns) - set(sectors.keys())
        if len(tmp) > 0:
            warnings.warn(f'{tmp} do not have sectors specified and was dropped')
        price_data = price_data[price_data.columns.intersection(sectors.keys())]

        # save arguments
        self.sectors = sectors
        self.price_data = price_data
        self.returns = self.price_data.pct_change()
        self.stddev = self.returns.std()

        # Laloux filter
        self.mesoscopic_filter()

        self.communities = None
        self.weight = None

    ##########################
    # Mesoscopic Correlation #
    ##########################

    def mesoscopic_decompose(self, start=None, end=None):
        """
        Decompose the correlation matrix eigen-spectrum into components from
        Random Noise, Mesoscopic, and Market modes following Laloux 1999.

        Reference
        ---------
        Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999).
        Noise Dressing of Financial Correlation Matrices.
        Physical Review Letters, 83(7), 1467–1470.
        https://doi.org/10.1103/PhysRevLett.83.1467

        Parameters
        ----------
        start : str, int, or None, optional
            Start index/date to consider. Defaults to start of DataFrame.
        end : str, int, or None, optional
            End index/date to consider. Defaults to end of DataFrame.

        Returns
        -------
        eigvals : np.ndarray of shape (N,)
            Eigenvalues of the correlation matrix.
        eigvecs : np.ndarray of shape (N, N)
            Corresponding eigenvectors of the correlation matrix.
        components : np.ndarray of str, shape (N,)
            Component labels for each eigenvalue and eigenvector.
        lambda_max : float
            The Marcenko–Pastur upper bound for random eigenvalues.
        lambda_1 : float
            The largest eigenvalue (market mode).
        """
        # get eigenspectrum of correlation matrix
        corr_matrix = self.returns[start:end].corr()
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)

        # market eigenvalue
        lambda_1 = np.max(eigvals)

        # Marcenko-Pastur distribution (w/o market)
        T, N = self.returns[start:end].shape
        lambda_max = (1 - lambda_1/N) * (1 + np.sqrt(N/T))**2

        # classify into components
        components = np.where(eigvals <= lambda_max, 'Random Noise',
            np.where(eigvals < lambda_1, 'Mesoscopic', 'Market'))

        return eigvals, eigvecs, components, lambda_max, lambda_1

    def mesoscopic_filter(self):
        """
        Apply the Laloux 1998 corrections to the correlation matrix, removing
        the Market and Random Noise components.

        Returns
        -------
        np.ndarray
            The mesoscopically-filtered correlation matrix.
        """
        # filter mesoscopic eigenvalue & eigenvectors
        eigvals, eigvecs, components, _, _ = self.mesoscopic_decompose()
        filt = (components == 'Mesoscopic')
        eigvals = eigvals[filt]
        eigvecs = eigvecs[:,filt]
        # reconstruct mesoscopic correlation & covariance
        self.corr_g =  eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.cov_g = np.diag(self.stddev) @ self.corr_g @ np.diag(self.stddev)

        return self.corr_g

    def cumulative_risk(self, start=None, end=None):
        """
        Calculate the cumulative risk attributed to each component

        Parameters
        ----------
        start : str, int, or None, optional
            Start index/date to consider. Defaults to start of DataFrame.
        end : str, int, or None, optional
            End index/date to consider. Defaults to end of DataFrame.

        Returns
        -------
        dict
            Maps component to cumulative risk
        """
        eigvals, _, components, _, _ = self.mesoscopic_decompose(start, end)
        risk = {label: eigvals[components == label].sum()
                for label in np.unique(components)}
        return risk

    def rolling_cumulative_risk(self, window=252, step=5, n_jobs=-1):
        """
        Compute the risk attributed to each spectral component over rolling
        windows of returns.

        Parameters
        ----------
        window : int, optional
            The number of trading days in each rolling window (default is 252).
        step : int, optional
            The stride with which the rolling window advances (default is 5).
        n_jobs : int, optional
            Number of parallel jobs to use (default is -1).

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by window start, with components as columns
        """
        idx = range(0, self.returns.shape[0] - window, step)
        with tqdm_joblib(tqdm(total=len(idx))) as progress_bar:
            data = Parallel(n_jobs=n_jobs)(
                delayed(self.cumulative_risk)(i, i+window)
                for i in idx
            )
        index = [self.returns.index[i] for i in idx]
        return pd.DataFrame(data, index=index)

    #######################
    # Community Detection #
    #######################

    @staticmethod
    def _commu_to_list(G, communities):
        """
        Convert a community assignment to a list of labels indexed by nodes.

        Parameters
        ----------
        G : networkx.Graph
            Asset graph.
        communities : list of lists
            Community structure as returned by community detection algorithms.

        Returns
        -------
        list of int
            List of community labels, index-aligned with nodes.
        """
        node_to_comm = {}
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                node_to_comm[node] = comm_id
        n_assets = len(G.nodes)
        communities = [node_to_comm[i] for i in range(n_assets)]
        return communities

    @staticmethod
    def kmeans_C_g(C_g, n_clusters):
        """
        Cluster assets using KMeans on the filtered correlation/covariance matrix.

        Parameters
        ----------
        C_g : np.ndarray
            Filtered correlation/covariance matrix.
        n_clusters : int
            Number of clusters to form.

        Returns
        -------
        np.ndarray
            Array of cluster labels.
        """
        labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(C_g)
        return labels

    @staticmethod
    def DBSCAN_Cg(C_g, returns, eps=0.5, min_samples=5, metric='precomputed'):
        """
        Cluster assets using DBSCAN on the mesoscopic representation.

        Parameters
        ----------
        C_g : np.ndarray
            Filtered correlation/covariance matrix.
        returns : pd.DataFrame or np.ndarray
            Asset returns matrix.
        eps : float, optional
            DBSCAN eps parameter. Default is 0.5.
        min_samples : int, optional
            DBSCAN min_samples parameter. Default is 5.
        metric : str, optional
            Metric for pairwise distance. Default is 'precomputed'.

        Returns
        -------
        np.ndarray
            Array of cluster labels.
        """
        X = (returns @ C_g).T
        dist = pairwise_distances(X, metric='correlation')
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(dist)
        return labels

    @staticmethod
    def sector_to_label(list_sector):
        """
        Map sector names to integer labels.

        Parameters
        ----------
        list_sector : list of str
            List of sectors.

        Returns
        -------
        list of int
            Corresponding integer labels for each sector.
        """
        labels, uniques = pd.factorize(list_sector)
        return labels.tolist()

    def community_discover(self, algo = "Louvain", seed = 42, **kwargs):
        """
        Discover asset communities using various algorithms.

        Parameters
        ----------
        algo : {'Louvain', 'Label', 'Kmean', 'DBSCAN', 'Sector'}, optional
            Algorithm to use for community detection. Default is "Louvain".
        seed : int, optional
            Random seed for community detection, where applicable.
        **kwargs :
            Additional keyword arguments for clustering algorithms

        Returns
        -------
        list or np.ndarray
            Community labels for each asset.

        Raises
        ------
        ValueError
            If the mesoscopic structure is not computed, or an algorithm name is invalid.
        """
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
        """
        Plot a pie chart breakdown of sector compositions within detected asset communities.

        Parameters
        ----------
        title : str, optional
            Title for the plot. Default is 'Louvain'.

        Raises
        ------
        ValueError
            If communities have not been discovered.
        """
        if self.communities is None:
            raise ValueError("No communities found, please run community_discover")
        comm2sectors = defaultdict(list)
        for asset_idx, comm_id in enumerate(self.communities):
            comm2sectors[comm_id].append(self.sector[asset_idx])

        sector_labels = sorted(set(self.sector))
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ffd700'
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
        """
        Compute the Global Minimum Variance (GMV) portfolio for a given covariance matrix.

        Parameters
        ----------
        cov_g : np.ndarray
            Asset covariance matrix.
        short : bool, optional
            If False, impose no short-selling (weights >= 0). Default is False.

        Returns
        -------
        np.ndarray
            Optimal portfolio weights.
        """
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
        """
        Compute the Global Minimum Variance (GMV) portfolio with community aggregation.

        Parameters
        ----------
        cov_g : np.ndarray
            Asset covariance matrix.
        community_labels : list or np.ndarray
            Community label for each asset.
        short : bool, optional
            If False, impose no short-selling (weights >= 0). Default is False.

        Returns
        -------
        np.ndarray
            Optimal portfolio weights, distributed equally within each community.
        """
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

    def portfolio_building(self, seed=42, **kwargs):
        """
        Construct the portfolio weights using selected methodology.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility in community detection. Default is 42.
        community : bool, optional
            If True, use community-based GMV optimization. Default is True.
        is_equal : bool, optional
            If True, use an equally-weighted portfolio. Default is False.
        short : bool, optional
            If False, impose no short-selling (weights >= 0). Default is False.
        algo : str, optional
            Community detection algorithm; passed to `community_discover`.
        min_samples : int, optional
            Used for DBSCAN. Default is 5.
        eps : float, optional
            Used for DBSCAN clustering.
        metric : str, optional
            Metric for DBSCAN clustering.

        Returns
        -------
        np.ndarray
            Portfolio weights.

        Raises
        ------
        ValueError
            If both `community` and `is_equal` are True, or if required data is missing.
        """
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