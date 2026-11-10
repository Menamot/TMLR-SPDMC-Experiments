import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from dbc.discretizers.base import BaseDiscretizer
from dbc.utils import _iter_batches


class GaussianMixtureDiscretizer(BaseDiscretizer):
    """Discretizer based on Gaussian Mixture Model (GMM).

    Each mixture component is treated as a profile.
    Soft membership (responsibilities) is used as degree.

    Attributes:
        centers_: Component means of shape (n_components, n_features), set after fit.
        gmm_: Fitted GaussianMixture instance, set after fit.
    """

    _is_soft = True

    def __init__(
        self,
        n_components: int = 5,
        covariance_type: str = "diag",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        random_state=None,
        max_iter: int = 100,
        batch_size: int = 16384,
    ):
        """Initializes GaussianMixtureDiscretizer.

        Args:
            n_components: Number of mixture components.
            covariance_type: Type of covariance parameters.
                One of ``{"full", "tied", "diag", "spherical"}``.
            tol: Convergence tolerance for EM.
            reg_covar: Regularization added to the diagonal of covariance matrices.
            random_state: Random seed for reproducibility.
            max_iter: Maximum number of EM iterations.
            batch_size: Number of samples per chunk when computing responsibilities
                in transform(). Larger values may be faster but use more memory.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """Fits a GMM on X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Ignored.

        Returns:
            self
        """
        X = validate_data(self, X, dtype=np.float64)
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.gmm_.fit(X)
        self.centers_ = self.gmm_.means_
        return self

    def transform(self, X):
        """Returns soft component responsibilities for each sample.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Responsibility matrix of shape (n_components, n_samples).
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        n_samples = X.shape[0]
        membership_degree = np.empty((self.n_components, n_samples), dtype=np.float64)
        for batch_start, batch_end in _iter_batches(n_samples, self.batch_size):
            membership_degree[:, batch_start:batch_end] = (
                self.gmm_.predict_proba(X[batch_start:batch_end]).T
            )
        return membership_degree


class GaussianMixtureClasswiseDiscretizer(BaseDiscretizer):
    """Class-conditional GMM discretization.

    Fits one GMM per class to model p(x | y=c). transform() returns joint
    responsibilities over (class, component) as a matrix of shape
    (sum_c K_c, n_samples), where responsibilities are normalized across all
    (class, component) pairs jointly.

    Attributes:
        classes_: Unique class labels seen during fit.
        class_priors_: Dict mapping class label to prior probability.
        log_class_priors_: Dict mapping class label to log prior probability.
        gmms_: Dict mapping class label to fitted GaussianMixture.
        component_slices_: Dict mapping class label to (start, end) index range
            into the responsibility rows.
        n_components_: Total number of components across all classes, set after fit.
    """

    _is_soft = True

    def __init__(
        self,
        n_components_per_class: int | dict = 4,
        covariance_type: str = "diag",
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state=None,
        use_empirical_priors: bool = True,
        batch_size: int = 16384,
    ):
        """Initializes GaussianMixtureClasswiseDiscretizer.

        Args:
            n_components_per_class: Number of mixture components per class. Pass
                an int to use the same count for all classes, or a dict mapping
                class label to component count for per-class control.
            covariance_type: Type of covariance parameters.
                One of ``{"full", "tied", "diag", "spherical"}``.
            reg_covar: Regularization added to the diagonal of covariance matrices.
            max_iter: Maximum number of EM iterations per class GMM.
            tol: Convergence tolerance for EM.
            random_state: Random seed for reproducibility.
            use_empirical_priors: If True, p(y=c) is estimated from training
                frequencies and included in the joint log-probability; otherwise
                uniform priors are used.
            batch_size: Number of samples per chunk when computing responsibilities
                in transform(). Larger values may be faster but use more memory.
        """
        self.n_components_per_class = n_components_per_class
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.use_empirical_priors = use_empirical_priors
        self.batch_size = batch_size

    def fit(self, X, y):
        """Fits one GMM per class.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            self
        """
        X = validate_data(self, X, dtype=np.float64)
        check_classification_targets(y)

        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes

        if self.use_empirical_priors:
            priors = counts.astype(float) / counts.sum()
        else:
            priors = np.full_like(counts, 1.0 / len(classes), dtype=float)
        log_priors = np.log(priors)
        self.class_priors_ = dict(zip(classes, priors))
        self.log_class_priors_ = dict(zip(classes, log_priors))

        self.gmms_ = {}
        self.component_slices_ = {}
        use_dict = isinstance(self.n_components_per_class, dict)

        slice_start = 0
        for class_label in classes:
            X_class = X[y == class_label]
            if X_class.shape[0] < 2:
                raise ValueError(
                    f"Class {class_label} has too few samples to fit a GMM."
                )

            if use_dict:
                if class_label not in self.n_components_per_class:
                    raise ValueError(
                        f"Missing n_components_per_class for class {class_label}."
                    )
                n_comp = int(self.n_components_per_class[class_label])
            else:
                n_comp = self.n_components_per_class

            if n_comp < 1:
                raise ValueError("n_components_per_class must be >= 1.")

            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                n_init=1,
            )
            gmm.fit(X_class)
            self.gmms_[class_label] = gmm

            slice_end = slice_start + n_comp
            self.component_slices_[class_label] = (slice_start, slice_end)
            slice_start = slice_end

        self.n_components_ = slice_start
        return self

    def transform(self, X):
        """Returns joint responsibilities across all (class, component) pairs.

        Responsibilities are computed from class-conditional log-weighted
        probabilities and normalized jointly across all components and classes.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Responsibility matrix of shape (total_components, n_samples), where
            total_components = sum of n_components_per_class over all classes.
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        n_samples = X.shape[0]

        membership_degree = np.empty((self.n_components_, n_samples), dtype=np.float64)

        for batch_start, batch_end in _iter_batches(n_samples, self.batch_size):
            X_batch = X[batch_start:batch_end]

            # Slices are contiguous and cover 0..n_components_, so np.empty is safe.
            log_joint_batch = np.empty(
                (X_batch.shape[0], self.n_components_), dtype=np.float64
            )

            for class_label in self.classes_:
                gmm = self.gmms_[class_label]
                comp_start, comp_end = self.component_slices_[class_label]
                # _estimate_weighted_log_prob returns log(pi_k * p(x|k)) per component,
                # needed for cross-class joint normalization (no public equivalent).
                log_joint_batch[:, comp_start:comp_end] = (
                    gmm._estimate_weighted_log_prob(X_batch)
                    + self.log_class_priors_[class_label]
                )

            log_norm_batch = logsumexp(log_joint_batch, axis=1, keepdims=True)
            resp_batch = np.exp(log_joint_batch - log_norm_batch)
            membership_degree[:, batch_start:batch_end] = resp_batch.T

        return membership_degree
