import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from dbc.utils import (
    compute_prior,
    compute_pz_given_y,
    compute_prior_best,
    compute_posterior,
)
from dbc.cyprien_code import compute_piStar


class DiscreteBayesianClassifier(BaseEstimator):
    """Discrete Bayesian Classifier (DBC) and minimax variant (DMC).

    Combines an external discretizer with Bayesian risk minimization.
    The discretizer maps input features to a (soft or hard) cluster membership
    matrix; the classifier then uses P(Z|Y) and the chosen loss matrix to
    make minimum-expected-risk decisions.

    When ``minmax=True`` the classifier operates as a DMC/SPDMC: it searches
    for a prediction prior that balances empirical class-conditional risks.

    Args:
        discretizer: A fitted or unfitted discretizer implementing
            ``fit(X, y)`` and ``transform(X)``, and exposing a boolean
            ``_is_soft`` attribute.
        loss_function: Either the string ``"01"`` for the standard 0-1 loss,
            or a numeric ndarray of shape (n_classes, n_classes).
        box: Optional parameter region passed to ``compute_piStar`` for the
            hard-discretizer minimax search.
        minmax: If True, search for the equalization prior after fitting.
        minmax_eps: Training risk-range tolerance for the prior search.
        minmax_alpha: Numerator of the multiclass harmonic step size.
        minmax_n_iter: Maximum number of multiclass prior updates.

    Attributes:
        membership_train_: Membership matrix P(Z|X_train) of shape
            (n_clusters, n_samples), set after ``fit``.
        p_hat_: Class-conditional cluster distribution P(Z|Y) of shape
            (n_classes, n_clusters), set after ``fit``.
        prior_: Empirical training class priors P(Y) of shape (n_classes,),
            set after ``fit``.
        prior_best_: Learned prediction prior of shape (n_classes,),
            set after ``fit`` when ``minmax=True``.
        label_encoder_: Fitted LabelEncoder mapping original labels to
            contiguous integers, set after ``fit``.
    """

    def __init__(
        self,
        discretizer,
        loss_function: str | np.ndarray = "01",
        box=None,
        minmax: bool = False,
        minmax_eps: float = 1e-2,
        minmax_alpha: float = 2.0,
        minmax_n_iter: int = 1000,
    ):
        self.minmax = minmax
        self.minmax_eps = minmax_eps
        self.minmax_alpha = minmax_alpha
        self.minmax_n_iter = minmax_n_iter
        self.box = box
        self.discretizer = discretizer
        self.loss_function = loss_function

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DiscreteBayesianClassifier":
        """Fits the classifier on training data.

        Encodes labels, resolves the loss matrix, fits the discretizer,
        estimates P(Z|Y), and (when ``minmax=True``) searches for the
        equalization prior.

        Args:
            X_train: Feature matrix of shape (n_samples, n_features).
            y_train: Class labels of shape (n_samples,).

        Returns:
            self
        """
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        n_classes = len(np.unique(y_encoded))

        # Resolve loss matrix
        if isinstance(self.loss_function, str) and self.loss_function == "01":
            self.loss_function_ = np.ones((n_classes, n_classes)) - np.eye(n_classes)
        elif isinstance(self.loss_function, np.ndarray):
            if (
                self.loss_function.ndim == 2
                and self.loss_function.shape[0] == n_classes
                and self.loss_function.shape[1] == n_classes
                and np.issubdtype(self.loss_function.dtype, np.number)
            ):
                self.loss_function_ = self.loss_function
            else:
                raise ValueError(
                    f"Invalid loss_function matrix: expected shape ({n_classes}, {n_classes}), "
                    f"got {self.loss_function.shape}"
                )
        else:
            raise ValueError("Unsupported loss_function; use '01' or an ndarray.")

        # Estimate priors and fit discretizer
        self.prior_ = compute_prior(y_encoded)
        self.discretizer.fit(X_train, y_encoded)
        self.membership_train_ = self.discretizer.transform(X_train)
        self.p_hat_ = compute_pz_given_y(self.membership_train_, y_encoded)

        # Optional: search for a risk-balancing prediction prior
        if self.minmax:
            # Hard and soft discretizers require different minimax methods:
            # hard uses the DMC search; soft uses risk-driven prior updates.
            if not self.discretizer._is_soft:
                self.prior_best_ = compute_piStar(
                    self.p_hat_,
                    y_encoded,
                    n_classes,
                    self.loss_function_,
                    N=1000,
                    Box=self.box,
                )[0]
            else:
                self.prior_best_, self.history_ = compute_prior_best(
                    y=y_encoded,
                    loss_function=self.loss_function_,
                    p_hat=self.p_hat_,
                    membership_degree=self.membership_train_,
                    prior_train=self.prior_,
                    eps=self.minmax_eps,
                    alpha=self.minmax_alpha,
                    n_iter=self.minmax_n_iter,
                    return_history=True,
                )
        return self

    def _compute_sample_conditional_risk(
        self, X: np.ndarray, prior_pred: np.ndarray | None = None
    ) -> np.ndarray:
        """Computes per-sample conditional risk for each possible decision.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            prior_pred: Prediction prior of shape (n_classes,). Defaults to
                ``prior_best_`` when ``minmax=True``, else ``prior_``.

        Returns:
            Risk matrix of shape (n_samples, n_classes).
        """
        check_is_fitted(self, ["p_hat_"])

        if prior_pred is None:
            prior_pred = self.prior_best_ if self.minmax else self.prior_

        posterior = compute_posterior(
            membership_pred=self.discretizer.transform(X),
            p_hat=self.p_hat_,
            prior_train=self.prior_,
            prior_pred=prior_pred,
        )
        return posterior @ self.loss_function_

    def predict(self, X: np.ndarray, prior_pred: np.ndarray | None = None) -> np.ndarray:
        """Predicts class labels by minimizing conditional risk.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            prior_pred: Optional override for the prediction prior.

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        sample_conditional_risk = self._compute_sample_conditional_risk(X, prior_pred)
        y_pred = np.argmin(sample_conditional_risk, axis=1)
        return self.label_encoder_.inverse_transform(y_pred)

    def predict_prob(self, X: np.ndarray, prior_pred: np.ndarray | None = None) -> np.ndarray:
        """Predicts class probabilities from normalized inverse risks.

        Converts the risk vector for each sample into a probability distribution
        by computing the complement of each risk relative to the total risk sum.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            prior_pred: Optional override for the prediction prior.

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        sample_conditional_risk = self._compute_sample_conditional_risk(X, prior_pred)
        # Convert risks to probabilities: higher risk → lower probability.
        # For each sample, p_k ∝ (sum_j risk_j) - risk_k.
        temp = np.sum(sample_conditional_risk, axis=1, keepdims=True) - sample_conditional_risk
        return temp / temp.sum(axis=1, keepdims=True)
