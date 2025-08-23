import numpy as np
import pandas as pd
import cvxpy as cp

import warnings
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from collections import Counter

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
        Price data with tickers as columns and time-steps as rows.
    sectors : dictionary
        Maps from ticker to sector.

    Attributes
    ----------
    price_data : pd.DataFrame
        As given in parameters
    sectors : dict
        As given in parameters
    returns : pd.DataFrame
        Percentage returns per time-step
    stddev : pd.Series
        Standard deviation of asset returns.
    corr : pd.DataFrame
        Correlation between asset returns, after Laloux's correction
    cov : pd.DataFrame
        Covariance between asset returns, after Laloux's correction
    communities : dict
        Mapping from ticker to detected community label.
    weights : pd.Series
        Optimized portfolio weights from the GMV portfolio construction.
    """
    def __init__(self, price_data, sectors, algo='Louvain'):
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
        self.corr, self.cov = self.mesoscopic_filter()
        self.communities = self.community_detection(algo)
        self.weights = self.gmv_portfolio()

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
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.corr =  pd.DataFrame(
            eigvecs @ np.diag(eigvals) @ eigvecs.T,
            index=tickers, columns=tickers
        )
        self.cov = pd.DataFrame(
            np.diag(self.stddev) @ self.corr.values @ np.diag(self.stddev),
            index=tickers, columns=tickers
        )

        return self.corr, self.cov

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

    def community_detection(self, algo='Louvain', **kwargs):
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
        dict
            Dictionary mapping asset ticker to identified community (0,1,2,...)
        """
        # Louvain community detection
        if algo == 'Louvain':
            kwargs['weight'] = 'weight'
            G = nx.from_numpy_array(self.corr.values)
            comm = louvain_communities(G, **kwargs)
            labels = self._communities_to_labels(comm, len(G.nodes))
            self.communities = dict(zip(self.corr.index, labels))
        # Label propagation community detection
        elif algo == 'Label Propagation':
            kwargs['weight'] = 'weight'
            G = nx.from_numpy_array(self.corr.values)
            comm = fast_label_propagation_communities(G, **kwargs)
            labels = self._communities_to_labels(comm, len(G.nodes))
            self.communities = dict(zip(self.corr.index, labels))
        # Agglomerative clustering
        elif algo == 'Agglomerative':
            kwargs['metric'] = 'precomputed'
            kwargs.setdefault('linkage', 'average')
            labels = AgglomerativeClustering(**kwargs).fit_predict(1 - self.corr.values)
            self.communities = dict(zip(self.corr.index, labels))
        # DBSCAN clustering
        elif algo == 'DBSCAN':
            kwargs['metric'] = 'precomputed'
            kwargs.setdefault('eps', 0.8)
            labels = DBSCAN(**kwargs).fit_predict(1 - self.corr.values)
            labels += 1 # -1 label lumped into 1 cl
            self.communities = dict(zip(self.corr.index, labels))
        # k-means clustering
        elif algo == 'Kmean':
            kwargs['n_clusters'] = 5
            labels = KMeans(**kwargs).fit_predict(self.corr.values)
            self.communities = dict(zip(self.corr.index, labels))
        # GICS sector
        elif algo == 'Sector':
            labels, _ = pd.factorize(np.array(list(self.sectors.values())))
            self.communities = dict(zip(self.corr.index, labels))
        elif algo == 'Equal-Weight':
            labels = [0,] * self.corr.shape[0]
            self.communities = dict(zip(self.corr.index, labels))
        # No communities
        elif algo == 'None':
            labels = range(self.corr.shape[0])
            self.communities = dict(zip(self.corr.index, labels))
        # else raise error
        else:
            raise ValueError('Select correct algorithm for community discover')

        self.gmv_portfolio() # re-optimise portfolio
        return self.communities

    ##########################
    # Portfolio Optimisation #
    ##########################

    def gmv_portfolio(self, short=False):
        """
        Compute the Global Minimum Variance (GMV) portfolio at the community level.
        This allocates equal weight within clusters, calculates the inter-cluster
        covariance, and optimises the cluster weights to minimise portfolio variance.

        Parameters
        ----------
        short : bool, optional, default=False
            Indicates if short-selling is allowed (weights may be negative).

        Returns
        -------
        dict
            Dictionary mapping identified community to GMV portfolio weight
        """
        # aggregate correlation matrices by sector
        wts = self.cov.columns.map(self.communities)
        tmp = np.eye(np.max(wts)+1)
        wts = np.array([tmp[i] for i in wts])
        corr_comm = wts.T @ self.cov.values @ wts

        # GMV optimise
        w = cp.Variable(corr_comm.shape[0])
        objective = cp.Minimize(cp.quad_form(w, corr_comm))
        if short:
            constraints = [cp.sum(w) == 1]
        else:
            constraints = [cp.sum(w) == 1,  w >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # save optimised weights, and convert back to assets
        ncomm = Counter(self.communities.values())
        weights = [w.value[i] / ncomm[i] for i in range(corr_comm.shape[0])]
        self.weights = dict(map(lambda x: (x[0], weights[x[1]]), self.communities.items()))
        return self.weights