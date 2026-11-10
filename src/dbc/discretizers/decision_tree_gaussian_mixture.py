import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted, validate_data

from dbc.discretizers.base import BaseDiscretizer
from dbc.utils import _iter_batches


def _resp_diag_batched(
    X,
    weights,
    means,
    diag_vars,
    batch_size: int = 8192,
):
    """Batched responsibilities for diagonal-covariance Gaussian mixture.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        weights: Mixture weights of shape (n_components,).
        means: Component means of shape (n_components, n_features).
        diag_vars: Diagonal variances of shape (n_components, n_features).
        batch_size: Number of samples per batch.

    Returns:
        Responsibility matrix of shape (n_components, n_samples).
    """
    X = np.asarray(X, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    diag_vars = np.asarray(diag_vars, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n_samples, n_features = X.shape
    n_components = means.shape[0]

    # Precompute mixture constants once across all batches
    inv_var = 1.0 / diag_vars                                       # (K, d)
    log_det = np.sum(np.log(2.0 * np.pi * diag_vars), axis=1)      # (K,)
    log_pi = np.log(weights + 1e-300)                               # (K,)

    # Precompute per-component terms for the quadratic form:
    # ||x - mu||^2_{Sigma^{-1}} = sum_j x_j^2/v_j - 2 x_j*mu_j/v_j + mu_j^2/v_j
    M = means * inv_var                                              # (K, d): mu/var
    C = np.sum(means * M, axis=1)                                   # (K,):   mu^T Sigma^{-1} mu

    resp_T = np.empty((n_components, n_samples), dtype=np.float64)

    for batch_start, batch_end in _iter_batches(n_samples, batch_size):
        X_batch = X[batch_start:batch_end]                          # (b, d)

        quad_x = (X_batch * X_batch) @ inv_var.T                   # (b, K): x^T Sigma^{-1} x
        cross_term = X_batch @ M.T                                  # (b, K): x^T Sigma^{-1} mu
        quad = quad_x - 2.0 * cross_term + C[None, :]              # (b, K)

        log_joint = log_pi[None, :] - 0.5 * (log_det[None, :] + quad)
        log_norm = logsumexp(log_joint, axis=1, keepdims=True)
        resp_T[:, batch_start:batch_end] = np.exp(log_joint - log_norm).T

    return resp_T


def _resp_full_batched(
    X,
    weights,
    means,
    chols,
    log_dets,
    batch_size: int = 2048,
):
    """Batched responsibilities for full-covariance Gaussian mixture.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        weights: Mixture weights of shape (n_components,).
        means: Component means of shape (n_components, n_features).
        chols: Lower-triangular Cholesky factors of shape
            (n_components, n_features, n_features).
        log_dets: Log-determinants log(det(Sigma_k)) of shape (n_components,).
        batch_size: Number of samples per batch.

    Returns:
        Responsibility matrix of shape (n_components, n_samples).
    """
    X = np.asarray(X, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    chols = np.asarray(chols, dtype=np.float64)
    log_dets = np.asarray(log_dets, dtype=np.float64)

    n_samples, n_features = X.shape
    n_components = means.shape[0]

    log_pi = np.log(weights + 1e-300)                       # (K,)
    const = n_features * np.log(2.0 * np.pi)

    resp_T = np.empty((n_components, n_samples), dtype=np.float64)

    for batch_start, batch_end in _iter_batches(n_samples, batch_size):
        X_batch = X[batch_start:batch_end]                  # (b, d)

        log_joint_batch = np.empty((batch_end - batch_start, n_components), dtype=np.float64)

        for k in range(n_components):
            diff = X_batch - means[k]                       # (b, d)
            # Triangular solve: L z = diff^T  →  z^T z = ||Sigma^{-1/2}(x-mu)||^2
            sol = solve_triangular(
                chols[k], diff.T, lower=True, check_finite=False, overwrite_b=False
            )                                               # (d, b)
            quad = np.sum(sol * sol, axis=0)                # (b,)
            log_joint_batch[:, k] = log_pi[k] - 0.5 * (const + log_dets[k] + quad)

        log_norm_batch = logsumexp(log_joint_batch, axis=1, keepdims=True)
        resp_T[:, batch_start:batch_end] = np.exp(log_joint_batch - log_norm_batch).T

    return resp_T


class TreeLeafGMMDiscretizer(BaseDiscretizer):
    """Decision-tree leaf Gaussian mixture discretizer (one component per leaf).

    Fits a decision tree, then models each leaf as a Gaussian component using
    the empirical mean and covariance of the samples reaching that leaf (no EM).
    transform() returns soft responsibilities of shape (n_leaves, n_samples).

    Attributes:
        model_: Fitted DecisionTreeClassifier, set after fit.
        weights_: Component weights of shape (n_components,), set after fit.
        means_: Component means of shape (n_components, n_features), set after fit.
        diag_vars_: Diagonal variances of shape (n_components, n_features), set
            after fit when covariance_type is ``"diag"``; otherwise None.
        covariances_: Full covariance matrices of shape
            (n_components, n_features, n_features), set after fit when
            covariance_type is ``"full"``; otherwise None.
        chols_: Cholesky factors of shape (n_components, n_features, n_features),
            set after fit when covariance_type is ``"full"``; otherwise None.
        log_dets_: Log-determinants of shape (n_components,), set after fit when
            covariance_type is ``"full"``; otherwise None.
    """

    _is_soft = True

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        reg: float = 1e-6,
        random_state=None,
        uniform_weights: bool = False,
        min_leaf_samples_for_component=None,
        covariance_type: str = "diag",
        chol_reg_retry: int = 3,
        chol_reg_multiplier: float = 10.0,
        batch_size: int = 16384,
    ):
        """Initializes TreeLeafGMMDiscretizer.

        Args:
            max_depth: Maximum depth of the decision tree.
            min_samples_leaf: Minimum number of samples required at a leaf node.
            reg: Regularization added to the diagonal of covariance matrices.
            random_state: Random seed for reproducibility.
            uniform_weights: If True, all components get equal weight;
                otherwise weights are proportional to leaf sample counts.
            min_leaf_samples_for_component: Leaves with fewer than this many
                samples are skipped. None means no minimum.
            covariance_type: Type of covariance estimate.
                One of ``{"diag", "full"}``.
            chol_reg_retry: Number of extra regularization attempts if Cholesky
                decomposition fails (full covariance only).
            chol_reg_multiplier: Factor by which regularization is multiplied on
                each retry (full covariance only).
            batch_size: Number of samples per chunk in transform().
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg = reg
        self.random_state = random_state
        self.uniform_weights = uniform_weights
        self.min_leaf_samples_for_component = min_leaf_samples_for_component
        self.covariance_type = covariance_type
        self.chol_reg_retry = chol_reg_retry
        self.chol_reg_multiplier = chol_reg_multiplier
        self.batch_size = batch_size

    def fit(self, X, y):
        """Fits the decision tree and computes per-leaf Gaussian components.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            self
        """
        X = validate_data(self, X, dtype=np.float64)
        y = np.asarray(y)

        if self.covariance_type not in ("full", "diag"):
            raise ValueError("covariance_type must be 'full' or 'diag'.")

        # Train decision tree
        self.model_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)

        leaf_assign = self.model_.apply(X)
        leaf_ids = np.unique(leaf_assign)

        n_samples, n_features = X.shape

        means = []
        counts = []

        if self.covariance_type == "diag":
            diag_vars = []
        else:
            covs = []
            chols = []
            log_dets = []

        for leaf_id in leaf_ids:
            idx = np.where(leaf_assign == leaf_id)[0]
            X_leaf = X[idx]
            nk = X_leaf.shape[0]

            if nk == 0:
                continue
            if (
                self.min_leaf_samples_for_component is not None
                and nk < self.min_leaf_samples_for_component
            ):
                continue

            mu = X_leaf.mean(axis=0)
            means.append(mu)
            counts.append(nk)

            diff = X_leaf - mu

            if self.covariance_type == "diag":
                var = (diff * diff).mean(axis=0) + self.reg
                diag_vars.append(var)
            else:
                cov_empirical = (diff.T @ diff) / nk

                # Cholesky with exponential regularization retry.
                # Each attempt reconstructs cov from the empirical estimate so
                # that the regularization schedule is clean: reg, reg*m, reg*m², …
                L = None
                reg_now = self.reg
                for _ in range(max(1, self.chol_reg_retry + 1)):
                    cov = cov_empirical + reg_now * np.eye(n_features)
                    try:
                        L = np.linalg.cholesky(cov)
                        break
                    except np.linalg.LinAlgError:
                        reg_now *= self.chol_reg_multiplier

                if L is None:
                    raise np.linalg.LinAlgError(
                        "Cholesky failed even after regularization retries."
                    )

                covs.append(cov)
                chols.append(L)
                log_dets.append(2.0 * np.sum(np.log(np.diag(L))))

        if len(means) == 0:
            raise ValueError("No valid leaf components after filtering.")

        self.means_ = np.asarray(means, dtype=np.float64)
        K = self.means_.shape[0]

        if self.uniform_weights:
            self.weights_ = np.full(K, 1.0 / K, dtype=np.float64)
        else:
            counts = np.asarray(counts, dtype=np.float64)
            self.weights_ = counts / counts.sum()

        if self.covariance_type == "diag":
            self.diag_vars_ = np.asarray(diag_vars, dtype=np.float64)
            self.covariances_ = None
            self.chols_ = None
            self.log_dets_ = None
        else:
            self.covariances_ = np.asarray(covs, dtype=np.float64)
            self.chols_ = np.asarray(chols, dtype=np.float64)
            self.log_dets_ = np.asarray(log_dets, dtype=np.float64)
            self.diag_vars_ = None

        return self

    def transform(self, X):
        """Returns soft leaf responsibilities for each sample.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Responsibility matrix of shape (n_components, n_samples).
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)

        if self.covariance_type == "diag":
            return _resp_diag_batched(
                X=X,
                weights=self.weights_,
                means=self.means_,
                diag_vars=self.diag_vars_,
                batch_size=self.batch_size,
            )

        # full covariance: use a smaller batch to keep memory manageable
        return _resp_full_batched(
            X=X,
            weights=self.weights_,
            means=self.means_,
            chols=self.chols_,
            log_dets=self.log_dets_,
            batch_size=max(1, self.batch_size // 4),
        )
