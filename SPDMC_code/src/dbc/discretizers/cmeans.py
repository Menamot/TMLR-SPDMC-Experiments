import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from dbc.discretizers.base import BaseDiscretizer
from dbc.utils import _iter_batches


class CMeansDiscretizer(BaseDiscretizer):
    """Soft discretizer based on Fuzzy C-Means.

    Centers are obtained via KMeans (default) or full FCM, or provided externally.
    Membership is computed in one shot from fixed centers using skfuzzy.
    transform() returns a soft membership matrix of shape (n_clusters, n_samples).

    Attributes:
        centers_: Cluster centers of shape (n_clusters, n_features), set after fit.
    """

    _is_soft = True

    def __init__(
        self,
        n_clusters: int = 8,
        fuzzifier: float = 1.5,
        tol: float = 1e-4,
        max_iter: int = 300,
        random_state=None,
        init="k-means++",
        use_kmeans: bool = True,
        batch_size: int = 8192,
    ):
        """Initializes CMeansDiscretizer.

        Args:
            n_clusters: Number of clusters. Ignored if centers is provided.
            fuzzifier: Fuzziness exponent m (must be > 1.0). Larger values produce
                softer memberships.
            tol: Convergence tolerance for FCM.
            max_iter: Maximum number of FCM iterations.
            random_state: Random seed for KMeans or FCM initialization.
            init: Pre-computed cluster centers of shape (n_clusters, n_features).
                If provided, fitting is skipped.
            use_kmeans: If True, use KMeans to obtain centers; otherwise run full FCM.
            batch_size: Number of samples per chunk when computing memberships in
                transform(). Reduces peak memory usage.
        """
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.use_kmeans = use_kmeans
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)

        if not isinstance(self.init, str):
            self.centers_ = check_array(self.init, dtype=np.float64, ensure_2d=True)
            self.n_clusters = self.centers_.shape[0]
            return self

        if self.use_kmeans:
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                random_state=self.random_state,
            )
            kmeans.fit(X)
            self.centers_ = kmeans.cluster_centers_
            return self

        self.centers_, *_ = fuzz.cluster.cmeans(
            X.T,
            c=self.n_clusters,
            m=self.fuzzifier,
            error=self.tol,
            maxiter=self.max_iter,
            seed=self.random_state,
        )
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        n_samples = X.shape[0]
        membership_degree = np.empty((self.n_clusters, n_samples), dtype=np.float64)
        for batch_start, batch_end in _iter_batches(n_samples, self.batch_size):
            membership_degree[:, batch_start:batch_end], *_ = (
                fuzz.cluster.cmeans_predict(
                    X[batch_start:batch_end].T,
                    cntr_trained=self.centers_,
                    m=self.fuzzifier,
                    error=self.tol,
                    maxiter=1,
                )
            )
        return membership_degree


class CMeansClasswiseDiscretizer(BaseDiscretizer):
    """Soft discretizer based on per-class Fuzzy C-Means clustering.

    Fits one FCM (or KMeans) model per class, concatenating all cluster centers.
    transform() returns a soft membership matrix of shape (n_clusters_total, n_samples).

    Attributes:
        centers_: All cluster centers of shape (n_clusters_total, n_features), set after fit.
        center_labels_: Class label for each center row, set after fit. None when
            centers is provided externally.
        n_clusters_: Total number of cluster centers, set after fit.
    """

    _is_soft = True

    def __init__(
        self,
        n_clusters_per_class: int | dict = 3,
        fuzzifier: float = 1.5,
        tol: float = 1e-4,
        max_iter: int = 300,
        random_state=None,
        init=None,
        use_kmeans: bool = False,
        batch_size: int = 8192,
    ):
        """Initializes CMeansClasswiseDiscretizer.

        Args:
            n_clusters_per_class: Number of clusters per class. Pass a dict mapping
                class label to cluster count for per-class control.
            fuzzifier: Fuzziness exponent m (must be > 1.0). Larger values produce
                softer memberships.
            tol: Convergence tolerance for FCM.
            max_iter: Maximum number of FCM iterations.
            random_state: Random seed for KMeans or FCM initialization.
            init: Pre-computed cluster centers of shape (n_clusters_total, n_features)
                across all classes. If provided, fitting is skipped.
            use_kmeans: If True, use KMeans instead of FCM to obtain per-class centers.
            batch_size: Number of samples per chunk when computing memberships in
                transform(). Use None to process all samples at once.
        """
        self.n_clusters_per_class = n_clusters_per_class
        self.fuzzifier = fuzzifier
        self.use_kmeans = use_kmeans
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X, y):
        X = validate_data(self, X, dtype=np.float64)
        check_classification_targets(y)

        if self.init is not None:
            self.centers_ = check_array(self.init, dtype=np.float64, ensure_2d=True)
            self.center_labels_ = None  # unknown when centers are provided externally
            self.n_clusters_ = self.centers_.shape[0]
            return self

        classes = np.unique(y)
        centers_all = []
        center_labels = []

        use_dict = isinstance(self.n_clusters_per_class, dict)

        for class_label in classes:
            X_class = X[y == class_label]

            if use_dict:
                n_clusters = self.n_clusters_per_class.get(int(class_label), 3)
            else:
                n_clusters = self.n_clusters_per_class

            if n_clusters == 1:
                center = X_class.mean(axis=0, keepdims=True)
            elif self.use_kmeans:
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                kmeans.fit(X_class)
                center = kmeans.cluster_centers_
            else:
                center, *_ = fuzz.cluster.cmeans(
                    X_class.T,
                    c=n_clusters,
                    m=self.fuzzifier,
                    error=self.tol,
                    maxiter=self.max_iter,
                    seed=self.random_state,
                )

            centers_all.append(center)
            center_labels.extend([class_label] * n_clusters)

        self.centers_ = np.vstack(centers_all)
        self.center_labels_ = np.array(center_labels)
        self.n_clusters_ = self.centers_.shape[0]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        n_samples = X.shape[0]
        membership_degree = np.empty((self.n_clusters_, n_samples), dtype=np.float64)
        for batch_start, batch_end in _iter_batches(n_samples, self.batch_size):
            membership_degree[:, batch_start:batch_end], *_ = (
                fuzz.cluster.cmeans_predict(
                    X[batch_start:batch_end].T,
                    cntr_trained=self.centers_,
                    m=self.fuzzifier,
                    error=self.tol,
                    maxiter=1,
                )
            )
        return membership_degree
