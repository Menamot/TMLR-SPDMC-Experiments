import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from dbc.discretizers.base import BaseDiscretizer
from dbc.utils import _iter_batches


def _assign_nearest_center(X, centers, batch_size):
    """Returns the index of the nearest center for each sample.

    Computes pairwise Euclidean distances and returns the argmin. When
    batch_size is set, processes samples in chunks to limit peak memory usage.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        centers: Center matrix of shape (n_centers, n_features).
        batch_size: Number of samples per chunk. If None, 0, or >= n_samples,
            all samples are processed at once.

    Returns:
        Integer array of shape (n_samples,) with the index of the nearest
        center for each sample.
    """
    n_samples = X.shape[0]
    nearest_indices = np.empty(n_samples, dtype=int)
    for batch_start, batch_end in _iter_batches(n_samples, batch_size):
        nearest_indices[batch_start:batch_end] = pairwise_distances_argmin(
            X[batch_start:batch_end], centers, metric="euclidean"
        )
    return nearest_indices


class KMeansDiscretizer(BaseDiscretizer):
    """Hard discretizer based on K-Means clustering.

    Each cluster centroid is treated as a discrete profile. transform() returns
    a one-hot membership matrix of shape (n_clusters, n_samples).

    Attributes:
        centers_: Cluster centers of shape (n_clusters, n_features), set after fit.
    """

    _is_soft = False

    def __init__(
        self,
        n_clusters: int = 8,
        init="k-means++",
        random_state=None,
        batch_size: int = 16384,
    ):
        """Initializes KMeansDiscretizer.

        Args:
            n_clusters: Number of clusters. Ignored when init is array-like.
            init: Initialization strategy. ``"k-means++"`` or ``"random"`` are
                passed to KMeans; an array-like of shape (n_clusters, n_features)
                is used directly as cluster centers without fitting.
            random_state: Random seed passed to KMeans.
            batch_size: Number of samples per chunk when assigning nearest centers.
                Use None to process all samples at once.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)
        if not isinstance(self.init, str):
            self.centers_ = check_array(self.init, dtype=np.float64, ensure_2d=True)
            self.n_clusters = self.centers_.shape[0]
        else:
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                random_state=self.random_state,
            )
            kmeans.fit(X)
            self.centers_ = kmeans.cluster_centers_
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        cluster_indices = _assign_nearest_center(X, self.centers_, self.batch_size)
        n_samples = X.shape[0]
        membership_degree = np.zeros((self.n_clusters, n_samples), dtype=np.float64)
        membership_degree[cluster_indices, np.arange(n_samples)] = 1.0
        return membership_degree


class KMeansClasswiseDiscretizer(BaseDiscretizer):
    """Hard discretizer based on per-class K-Means clustering.

    Fits one KMeans model per class and concatenates all cluster centers,
    allowing the discretizer to capture class-specific structure. transform()
    returns a one-hot membership matrix of shape (n_clusters_total, n_samples).

    Attributes:
        centers_: All cluster centers of shape (n_clusters_total, n_features), set after fit.
        center_labels_: Class label for each center row, set after fit. None when
            init is provided as array-like.
        n_clusters_: Total number of cluster centers, set after fit.
    """

    _is_soft = False

    def __init__(
        self,
        n_clusters_per_class: int | dict = 3,
        init="k-means++",
        random_state=None,
        batch_size: int = 16384,
    ):
        """Initializes KMeansClasswiseDiscretizer.

        Args:
            n_clusters_per_class: Number of clusters per class. Pass a dict
                mapping class label to cluster count for per-class control.
            init: Initialization strategy passed to each per-class KMeans call
                (``"k-means++"`` or ``"random"``), or an array-like of shape
                (n_clusters_total, n_features) to use as centers directly without fitting.
            random_state: Random seed passed to KMeans.
            batch_size: Number of samples per chunk when assigning nearest centers.
                Use None to process all samples at once.
        """
        self.n_clusters_per_class = n_clusters_per_class
        self.init = init
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X, y):
        X = validate_data(self, X, dtype=np.float64)
        check_classification_targets(y)

        if not isinstance(self.init, str):
            self.centers_ = check_array(self.init, dtype=np.float64, ensure_2d=True)
            self.center_labels_ = None  # unknown when centers are provided externally
            self.n_clusters_ = self.centers_.shape[0]
            return self

        classes = np.unique(y)
        centers_all = []
        center_labels = []
        use_dict = isinstance(self.n_clusters_per_class, dict)

        for class_label in classes:
            if use_dict:
                n_clusters = int(self.n_clusters_per_class.get(int(class_label), 3))
            else:
                n_clusters = self.n_clusters_per_class

            X_class = X[y == class_label]
            if X_class.shape[0] == 0:
                raise ValueError("No instances in some classes.")

            if n_clusters == 1:
                centers = X_class.mean(axis=0, keepdims=True)
            else:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    init=self.init,
                    random_state=self.random_state,
                )
                kmeans.fit(X_class)
                centers = kmeans.cluster_centers_

            centers_all.append(centers)
            center_labels.extend([class_label] * n_clusters)

        if len(centers_all) == 0:
            raise ValueError(
                "No centers were created. Check your labels and n_clusters_per_class."
            )

        self.centers_ = np.vstack(centers_all)
        self.center_labels_ = np.asarray(center_labels)
        self.n_clusters_ = self.centers_.shape[0]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        cluster_indices = _assign_nearest_center(X, self.centers_, self.batch_size)
        n_samples = X.shape[0]
        membership_degree = np.zeros((self.n_clusters_, n_samples), dtype=np.float64)
        membership_degree[cluster_indices, np.arange(n_samples)] = 1.0
        return membership_degree
