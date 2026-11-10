"""Randomized hard-profile SoftminDMC of Gilet et al., Machine Learning (2024).

DOI: 10.1007/s10994-023-06397-8. This is distinct from soft discretization.
Probabilities describe randomized actions, not calibrated class posteriors.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from .discretizers.kmeans import KMeansDiscretizer
from .utils import compute_pz_given_y, proj_simplex_Condat


@dataclass(frozen=True)
class SoftminPriorResult:
    prior: np.ndarray
    residual: float
    class_risks: np.ndarray
    n_iter: int
    best_iteration: int
    converged: bool


def profile_action_probabilities(p_hat, loss, prior, softmin_lambda):
    """Return Q[action, profile]; deliberately do NOT divide by P(Z)."""
    costs = np.asarray(loss).T @ (np.asarray(prior)[:, None] * p_hat)
    return softmax(-softmin_lambda * costs, axis=0)


def softmin_class_risks(p_hat, loss, action_probabilities):
    """Internal expected class risks, used only for the prior optimization."""
    return np.sum(p_hat * (loss @ action_probabilities), axis=1)


def optimize_softmin_prior(p_hat, loss, softmin_lambda, *, max_iter=50000, tol=1e-5):
    """Projected +G update with harmonic steps, uniform start, best residual.

    max_iter counts updates; the initial and final iterate are both evaluated.
    A finite nonconverged result is returned with converged=False. No general
    convergence guarantee or randomized-risk certificate is asserted.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    loss = np.asarray(loss, dtype=float)
    if (p_hat.ndim != 2 or min(p_hat.shape) == 0
            or not np.isfinite(p_hat).all() or np.any(p_hat < 0)
            or not np.allclose(p_hat.sum(axis=1), 1)):
        raise ValueError("p_hat must contain one probability distribution per class")
    k = p_hat.shape[0]
    if loss.shape != (k, k) or not np.isfinite(loss).all() or np.any(loss < 0):
        raise ValueError("loss must be a finite nonnegative class-by-action matrix")
    if not np.isfinite(softmin_lambda) or softmin_lambda <= 0:
        raise ValueError("softmin_lambda must be finite and positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    pi = np.full(k, 1 / k)
    best_residual = np.inf
    for iteration in range(max_iter + 1):
        q = profile_action_probabilities(p_hat, loss, pi, softmin_lambda)
        risks = softmin_class_risks(p_hat, loss, q)
        g = risks - float(pi @ risks)
        residual = float(np.linalg.norm(g))
        if not np.isfinite(residual):
            raise FloatingPointError("Non-finite SoftminDMC optimization residual")
        if residual < best_residual:
            best_residual = residual
            best_pi, best_risks, best_iteration = pi.copy(), risks.copy(), iteration
        if best_residual <= tol or iteration == max_iter:
            break
        pi = proj_simplex_Condat(k, pi + g / ((iteration + 1) * max(1., residual)))
    return SoftminPriorResult(best_pi, best_residual, best_risks, iteration,
                              best_iteration, best_residual <= tol)


def sample_action_indices(probabilities, *, random_state=None, uniforms=None):
    """Inverse-CDF draws; supplied uniforms allow paired candidate comparisons."""
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.ndim != 2 or probabilities.shape[1] < 1
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Expected normalized sample-by-action probabilities")
    if uniforms is None:
        uniforms = np.random.default_rng(random_state).random(len(probabilities))
    uniforms = np.asarray(uniforms)
    if (uniforms.shape != (len(probabilities),) or not np.isfinite(uniforms).all()
            or np.any(uniforms < 0) or np.any(uniforms >= 1)):
        raise ValueError("uniforms must contain one value in [0, 1) per sample")
    cdf = probabilities.cumsum(axis=1)
    cdf[:, -1] = 1.
    return (uniforms[:, None] >= cdf).sum(axis=1)


class SoftminDiscreteMinimaxClassifier(ClassifierMixin, BaseEstimator):
    """Hard partition followed by learned-prior, randomized softmin decisions.

    loss_matrix rows follow sorted classes_ (truth), columns follow classes_
    (actions). The supplied discretizer is cloned. A fixed random_state makes
    predict repeatable for the same input order without touching global RNG.
    """

    def __init__(self, discretizer=None, *, softmin_lambda=120., loss_matrix=None,
                 max_iter=50000, tol=1e-5, random_state=None):
        self.discretizer = discretizer
        self.softmin_lambda = softmin_lambda
        self.loss_matrix = loss_matrix
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _membership(self, X):
        membership = np.asarray(self.discretizer_.transform(X), dtype=float)
        if (membership.ndim != 2 or membership.shape[1] != len(X)
                or not np.all((membership == 0) | (membership == 1))
                or not np.all(membership.sum(axis=0) == 1)):
            raise ValueError("SoftminDMC requires one-hot hard-profile memberships")
        return membership

    def _fit_partition(self, X, y):
        """Internal split fit for tuning lambda while reusing the hard partition."""
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) < 2:
            raise ValueError("Training requires at least two classes")
        y_encoded = self.label_encoder_.transform(y)
        base = self.discretizer
        if base is None:
            base = KMeansDiscretizer(random_state=self.random_state)
        if getattr(base, '_is_soft', None) is not False:
            raise ValueError("SoftminDMC requires a hard discretizer")
        self.discretizer_ = clone(base).fit(X, y_encoded)
        self.p_hat_ = compute_pz_given_y(self._membership(X), y_encoded)
        k = len(self.classes_)
        self.loss_matrix_ = (np.ones((k, k)) - np.eye(k) if self.loss_matrix is None
                             else np.asarray(self.loss_matrix, dtype=float).copy())
        self.prior_train_ = np.bincount(y_encoded) / len(y_encoded)
        return self

    def _fit_prior(self):
        self.optimization_result_ = optimize_softmin_prior(
            self.p_hat_, self.loss_matrix_, self.softmin_lambda,
            max_iter=self.max_iter, tol=self.tol)
        result = self.optimization_result_
        self.prior_best_ = result.prior.copy()
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        self.residual_ = result.residual
        self.action_probabilities_ = profile_action_probabilities(
            self.p_hat_, self.loss_matrix_, self.prior_best_, self.softmin_lambda)
        return self

    def fit(self, X, y):
        return self._fit_partition(X, y)._fit_prior()

    def predict_proba(self, X):
        check_is_fitted(self, 'action_probabilities_')
        X = validate_data(self, X, dtype=np.float64, reset=False)
        profile_ids = self._membership(X).argmax(axis=0)
        return self.action_probabilities_[:, profile_ids].T

    def predict(self, X, *, random_state=None):
        seed = self.random_state if random_state is None else random_state
        indices = sample_action_indices(self.predict_proba(X), random_state=seed)
        return self.classes_[indices]
