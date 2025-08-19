import numpy as np
import pandas as pd
import cvxpy as cp

import warnings
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms.community import louvain_communities, fast_label_propagation_communities
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
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
        self.community_detection()
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
        tickers = self.returns.columns
        corr_g = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.corr_g =  pd.DataFrame(
            eigvecs @ np.diag(eigvals) @ eigvecs.T, 
            index=tickers, columns=tickers
        )
        self.cov_g = pd.DataFrame(
            np.diag(self.stddev) @ self.corr_g @ np.diag(self.stddev),
            index=tickers, columns=tickers
        )

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
    def _communities_to_labels(communities, n_nodes):
        """
        Convert community assignments into a node-indexed label list.
        """
        labels = [-1,] * n_nodes
        for i, comm in enumerate(communities):
            for node in comm: labels[node] = i
        return labels

    def community_detection(self, algo="Louvain", **kwargs):
        """
        Detect asset communities using various algorithms.

        Parameters
        ----------
        algo : {'Louvain','Label','Agglo','DBSCAN','Kmean','Sector'}, optional
            Algorithm to use for community detection. Default is "Louvain".
        **kwargs :
            Additional keyword arguments for each algorithm above

        Returns
        -------
        list or np.ndarray
            Community labels for each asset.
        """
        # Louvain community detection
        if algo == 'Louvain':
            kwargs['weight'] = 'weight'
            G = nx.from_numpy_array(self.corr_g.values)
            comm = louvain_communities(G, **kwargs)
            labels = self._communities_to_labels(comm, len(G.nodes))
            self.communities = dict(zip(self.corr_g.index, labels))
        # Label propagation community detection
        elif algo == 'Label Propagation':
            kwargs['weight'] = 'weight'
            G = nx.from_numpy_array(self.corr_g.values)
            comm = fast_label_propagation_communities(G, **kwargs)
            labels = self._communities_to_labels(comm, len(G.nodes))
            self.communities = dict(zip(self.corr_g.index, labels))
        # Agglomerative clustering
        elif algo == 'Agglomerative':
            kwargs['metric'] = 'precomputed'
            kwargs.setdefault('linkage', 'average')
            labels = AgglomerativeClustering(**kwargs).fit_predict(1 - self.corr_g.values)
            self.communities = dict(zip(self.corr_g.index, labels))
        # DBSCAN clustering
        elif algo == 'DBSCAN':
            kwargs['metric'] = 'precomputed'
            kwargs.setdefault('eps', 0.8)
            labels = DBSCAN(**kwargs).fit_predict(1 - self.corr_g.values)
            labels = [x if x != -1 else None for x in labels]
            self.communities = dict(zip(self.corr_g.index, labels))
        # k-means clustering
        elif algo == 'Kmean':
            kwargs['n_clusters'] = 5
            labels = KMeans(**kwargs).fit_predict(self.corr_g.values)
            self.communities = dict(zip(self.corr_g.index, labels))
        # GICS sector
        elif algo == 'Sector':
            labels, _ = pd.factorize(self.sector)
            self.communities = dict(zip(self.corr_g.index, labels))
        # else raise error
        else:
            raise ValueError('Select correct algorithm for community discover')

        return self.communities

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
            Community detection algorithm; passed to `community_detection`.
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
            self.communities = self.community_detection(algo = algo, seed = seed, **kwargs)
            self.w = self.solve_gmv_community(self.cov_g,self.communities, short = short)
        elif is_equal:
             self.w = np.ones(len(self.returns.columns)) * 1 / len(self.returns.columns)
        else:
            self.cov_g = self.returns.cov().values
            self.w = self.solve_gmv(self.cov_g, short)
        return self.w