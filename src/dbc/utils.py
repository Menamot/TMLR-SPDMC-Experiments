import numpy as np
import warnings
from typing import Generator, Tuple
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def _iter_batches(
    n_samples: int, batch_size: int | None
) -> Generator[Tuple[int, int], None, None]:
    """Yields (start, end) index pairs for iterating over samples in batches.

    Args:
        n_samples: Total number of samples.
        batch_size: Number of samples per batch. If ``None``, ``0``, or
            ``>= n_samples``, yields a single ``(0, n_samples)`` pair so
            callers can be written as a single unconditional loop.

    Yields:
        Tuple of ``(batch_start, batch_end)`` integer indices.
    """
    if batch_size is None or batch_size <= 0 or batch_size >= n_samples:
        yield 0, n_samples
        return
    for start in range(0, n_samples, batch_size):
        yield start, min(start + batch_size, n_samples)


def compute_pz_given_y(membership: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the class-conditional cluster distribution P(Z|Y).

    Args:
        membership: Membership matrix of shape (n_clusters, n_samples).
            Each column is the membership vector for one sample.
        y: Integer class labels of shape (n_samples,), assumed to be
            contiguous in the range [0, n_classes).

    Returns:
        Matrix of shape (n_classes, n_clusters) where entry [k, t] is
        the average membership of cluster t over samples of class k.
    """
    n_classes = int(y.max()) + 1
    class_counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    # one_hot: (n_samples, n_classes) — indicator matrix for class membership
    one_hot = np.zeros((len(y), n_classes), dtype=np.float64)
    one_hot[np.arange(len(y)), y] = 1.0
    # (n_clusters, n_samples) @ (n_samples, n_classes) -> (n_clusters, n_classes), then transpose
    pz_given_y = (membership @ one_hot).T  # (n_classes, n_clusters)
    pz_given_y /= class_counts[:, None]
    return pz_given_y


def compute_pz(p_hat: np.ndarray, prior_train: np.ndarray) -> np.ndarray:
    """Computes the marginal cluster distribution P(Z) = sum_k P(Z|Y=k) * P(Y=k).

    Args:
        p_hat: Conditional distribution P(Z|Y) of shape (n_classes, n_clusters).
        prior_train: Class prior probabilities P(Y) of shape (n_classes,).

    Returns:
        Marginal distribution P(Z) of shape (n_clusters,).
    """
    pz = np.sum(prior_train.reshape(-1, 1) * p_hat, axis=0)
    if np.min(pz) < 1e-12:
        # A near-zero P(Z) component means a cluster is essentially out-of-distribution;
        # downstream division by P(Z) will be numerically unstable.
        warnings.warn(
            "Some components of P(Z) are extremely small (< 1e-12). "
            "This may cause numerical instability in later computations.",
            RuntimeWarning,
        )
    return pz


def compute_prior(y: np.ndarray) -> np.ndarray:
    """Computes empirical class prior probabilities from labels.

    Args:
        y: Integer class labels of shape (n_samples,), assumed contiguous
            in the range [0, n_classes).

    Returns:
        Prior probability array of shape (n_classes,).
    """
    counts = np.bincount(y)
    return counts / counts.sum()


def compute_class_conditional_risk_with_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_function: np.ndarray = None,
    classes: np.ndarray = None,
):
    """Computes class-conditional risk from predicted labels.

    Convenience wrapper that builds a normalized confusion matrix and applies
    the loss function to obtain per-class average risks. Useful for evaluating
    a classifier's risk on a held-out set when only hard predictions are available.

    Args:
        y_true: Ground-truth class labels of shape (n_samples,).
        y_pred: Predicted class labels of shape (n_samples,).
        loss_function: Loss matrix of shape (n_classes, n_classes). Defaults to
            the 0-1 loss (identity subtracted from all-ones matrix).
        classes: Ordered array of class labels that fixes the output dimension and
            ordering. Defaults to ``np.unique(y_true)``.

    Returns:
        Tuple of:
            R: Class-conditional risk array of shape (n_classes,).
            classes: Class label array of shape (n_classes,).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if classes is None:
        classes = np.unique(y_true)

    # Fixed label space
    le = LabelEncoder()
    le.fit(classes)

    y_true_enc = le.transform(y_true)
    y_pred_enc = le.transform(y_pred)

    C = len(classes)
    if loss_function is None:
        loss_function = np.ones((C, C)) - np.eye(C)

    # Row-normalized confusion matrix P(pred=j | true=i)
    cm_norm = confusion_matrix(
        y_true_enc,
        y_pred_enc,
        labels=np.arange(C),
        normalize="true",
    )

    # R_i = sum_j L_{i,j} * P(pred=j | true=i)
    R = np.sum(loss_function * cm_norm, axis=1)
    return R, classes


def compute_posterior(
    membership_pred: np.ndarray,
    p_hat: np.ndarray,
    prior_train: np.ndarray,
    prior_pred: np.ndarray,
) -> np.ndarray:
    """Computes posterior class probabilities P(Y|X) for each sample.

    Uses Bayes' theorem: P(Y=k|X) ∝ P(Z|Y=k) * P(Y=k) / P(Z), where
    P(Z|X) is given by the membership vector.

    Args:
        membership_pred: Membership matrix of shape (n_clusters, n_samples).
        p_hat: Class-conditional cluster distribution P(Z|Y) of shape
            (n_classes, n_clusters).
        prior_train: Training class priors P(Y) used to compute P(Z),
            shape (n_classes,).
        prior_pred: Prediction class priors used for posterior weighting,
            shape (n_classes,).

    Returns:
        Posterior probability matrix of shape (n_samples, n_classes).
    """
    pz = compute_pz(p_hat, prior_train)
    diag_matrix = np.diag(1.0 / pz)
    prob = (prior_pred.reshape(-1, 1) * (p_hat @ (diag_matrix @ membership_pred))).T
    return prob


def compute_class_conditional_risk(
    y: np.ndarray,
    loss_function: np.ndarray,
    p_hat: np.ndarray,
    membership: np.ndarray,
    prior_train: np.ndarray,
    prior_pred: np.ndarray,
) -> np.ndarray:
    """Computes class-conditional risk for the soft DBC (vectorized).

    For each sample, finds the minimum-risk decision under the given prior and
    loss matrix, then averages the incurred losses per true class.

    Args:
        y: True class labels of shape (n_samples,).
        loss_function: Loss matrix L[y_true, y_pred] of shape
            (n_classes, n_classes).
        p_hat: Class-conditional cluster distribution P(Z|Y) of shape
            (n_classes, n_clusters).
        membership: Membership matrix P(Z|X) of shape (n_clusters, n_samples).
        prior_train: Training class priors P(Y) of shape (n_classes,), used
            to compute P(Z).
        prior_pred: Prediction priors of shape (n_classes,) used for risk
            weighting during decision making.

    Returns:
        Class-conditional risk array of shape (n_classes,).
    """
    # ----- shapes -----
    # n_classes := n_classes, T := n_clusters, S := n_samples
    # loss_function: (n_classes, n_classes)
    # p_hat:         (n_classes, n_clusters)
    # Pz_given_x:    (n_clusters, n_samples)
    # prior_train:   (n_classes,)
    # prior_pred:    (n_classes,)
    n_classes = loss_function.shape[0]

    # Class counts for averaging per class
    n_class_k = np.bincount(y, minlength=n_classes)

    # P(Z) using provided helper
    pz = compute_pz(p_hat, prior_train)  # (T,)

    # Build M = L[:, :, None] * prior_pred[:, None, None] * (p_hat[:, None, :] / pz)
    # Shape: (n_classes, n_classes, n_clusters)
    M = (
        loss_function[:, :, None]
        * prior_pred[:, None, None]
        * (p_hat[:, None, :] / pz[None, None, :])
    )

    # We need, for each sample with membership v (T,),  lambd = sum_{i,k} M[i, :, k] * v[k]
    # This equals (sum over i of M)[ :, :] @ v
    # Pre-sum over true-class axis i -> M_sum has shape (C_pred, T)
    M_sum = M.sum(axis=0)  # (n_classes, T)

    # For all samples in one shot:
    # Pz_given_x is (T, S).  Lambda_all = (M_sum @ Pz_given_x)^\top -> (S, n_classes)
    Lambda_all = (M_sum @ membership).T  # (S, n_classes)

    # For each sample, choose predicted class that minimizes lambda
    l_min_idx = np.argmin(Lambda_all, axis=1)  # (S,)

    # Per-sample incurred loss is L[y_true, y_pred*] where y_pred* = argmin_j lambda_j
    sample_losses = loss_function[y, l_min_idx]  # (S,)

    # Average per true class: sum of losses within class / number of samples in class
    # Use bincount with weights to aggregate per-class sums
    sum_losses_per_class = np.bincount(y, weights=sample_losses, minlength=n_classes)

    # Avoid divide-by-zero for classes absent in y
    with np.errstate(invalid="ignore", divide="ignore"):
        class_conditional_risk = np.where(
            n_class_k > 0,
            sum_losses_per_class / n_class_k,
            0.0,
        )

    return class_conditional_risk


def proj_simplex_Condat(n_classes: int, prob: np.ndarray) -> np.ndarray:
    """Projects a vector onto the probability simplex.

    Implements the O(n log n) algorithm from:
    L. Condat, "Fast projection onto the simplex and the ball",
    Mathematical Programming, vol. 158, no. 1, pp. 575-585, 2016.

    Args:
        n_classes: Dimension of the simplex (number of classes).
        prob: Vector of shape (n_classes,) to project.

    Returns:
        Projected vector of shape (n_classes,) satisfying sum = 1, entries >= 0.
    """
    linK = np.linspace(1, n_classes, n_classes)
    pi_proj = np.maximum(
        prob - np.max((np.cumsum(np.sort(prob)[::-1]) - 1) / linK), 0
    )
    return pi_proj / pi_proj.sum()


def projected_prior_update(risk_function, initial_prior, *, alpha=2.0,
                           n_iter=1000, eps=1e-3):
    """Algorithm 1: projected risk deviations with step alpha / (s + 1).

    The tolerance applies to sum((R - mean(R)) ** 2), evaluated after each
    update. Return the final prior and the initial/post-update risk trajectory.
    This heuristic can oscillate; reaching the iteration limit is not evidence
    of convergence.
    """
    pi = np.asarray(initial_prior, dtype=float).copy()
    if pi.ndim != 1 or len(pi) < 2 or not np.isfinite(pi).all():
        raise ValueError("initial_prior must be a finite probability vector")
    if np.any(pi < 0) or not np.isclose(pi.sum(), 1):
        raise ValueError("initial_prior must be nonnegative and sum to one")
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(eps) or eps < 0:
        raise ValueError("alpha must be positive and eps nonnegative")
    if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    def evaluate(prior):
        risk = np.asarray(risk_function(prior), dtype=float)
        if risk.shape != pi.shape or not np.isfinite(risk).all():
            raise ValueError("risk_function must return one finite risk per class")
        return risk
    risks = [evaluate(pi)]
    for step in range(n_iter):
        direction = risks[-1] - risks[-1].mean()
        pi = proj_simplex_Condat(len(pi), pi + alpha / (step + 1) * direction)
        risks.append(evaluate(pi))
        deviation = risks[-1] - risks[-1].mean()
        if float(deviation @ deviation) < eps:
            break
    return pi, np.asarray(risks)


def compute_prior_best(
    y: np.ndarray,
    loss_function: np.ndarray,
    p_hat: np.ndarray,
    membership_degree: np.ndarray,
    prior_train: np.ndarray,
    alpha: float = 2,
    n_iter: int = 1000,
    eps: float = 1e-2,
    return_history: bool = False,
):
    """Search for a prior with balanced empirical class-conditional risks.

    Binary problems use bisection. Multiclass problems use projected risk
    deviations with step alpha / (s + 1). eps is the risk-range tolerance;
    return_history includes the evaluated risk ranges. The multiclass update
    is heuristic and may reach the iteration limit without equalizing risks.
    """
    prior_best = prior_train.copy()
    n_classes = loss_function.shape[0]
    if n_classes == 2:
        return compute_prior_best_binary(
            y=y,
            loss_function=loss_function,
            p_hat=p_hat,
            membership=membership_degree,
            prior_train=prior_train,
            eps=eps,
            return_history=return_history,
        )
    history = []

    for n in range(n_iter):
        class_conditional_risk = compute_class_conditional_risk(
            y=y,
            loss_function=loss_function,
            p_hat=p_hat,
            membership=membership_degree,
            prior_train=prior_train,
            prior_pred=prior_best,
        )

        global_risk = np.mean(class_conditional_risk)
        G = class_conditional_risk - global_risk  # risk-deviation direction

        max_gap = np.max(class_conditional_risk) - np.min(class_conditional_risk)
        history.append(max_gap)
        if max_gap < eps:
            break

        w = prior_best + alpha / (n + 1) * G
        prior_best = proj_simplex_Condat(n_classes, w)

    if return_history:
        return prior_best, history
    else:
        return prior_best


def compute_prior_best_binary(
    y: np.ndarray,
    loss_function: np.ndarray,
    p_hat: np.ndarray,
    membership: np.ndarray,
    prior_train: np.ndarray,
    n_iter: int = 100,
    eps: float = 1e-3,
    return_history: bool = False,
    prior_update_eps: float = 1e-5,
):
    """Searches for a binary risk-balancing prior via bisection.

    For binary problems the risk-gap function R_0(pi) - R_1(pi) is monotone in
    pi_0, and bisection narrows the interval around a risk crossing.
    Finite-sample risks may jump across zero without exact equalization.

    Args:
        y: Integer class labels of shape (n_samples,).
        loss_function: Loss matrix of shape (2, 2).
        p_hat: Class-conditional cluster distribution P(Z|Y) of shape
            (2, n_clusters).
        membership: Membership matrix of shape (n_clusters, n_samples).
        prior_train: Training class priors of shape (2,).
        n_iter: Maximum number of bisection steps.
        eps: Convergence threshold on the absolute risk gap.
        return_history: If True, also returns a list of per-iteration gap values.
        prior_update_eps: Stop early when the bisection interval width falls
            below this value.

    Returns:
        prior_best: Optimized prior of shape (2,).
        history (optional): List of gap values per iteration, returned only
            when ``return_history=True``.
    """
    def compute_risk_gap(prior_pred: np.ndarray) -> float:
        class_conditional_risk = compute_class_conditional_risk(
            y=y,
            loss_function=loss_function,
            p_hat=p_hat,
            membership=membership,
            prior_train=prior_train,
            prior_pred=prior_pred,
        )
        return class_conditional_risk[0] - class_conditional_risk[1]

    # Evaluate at the two simplex extremes: pi_0 = 0 and pi_0 = 1
    prior_lower = np.array([0.0, 1.0])
    prior_upper = np.array([1.0, 0.0])
    gap_lower = compute_risk_gap(prior_lower)
    gap_upper = compute_risk_gap(prior_upper)

    history = []

    # Start with the endpoint that has the smaller absolute gap
    if abs(gap_lower) <= abs(gap_upper):
        prior_best, gap_best = prior_lower, gap_lower
    else:
        prior_best, gap_best = prior_upper, gap_upper

    for _ in range(n_iter):
        pi_mid = 0.5 * (prior_lower + prior_upper)
        gap_mid = compute_risk_gap(pi_mid)

        if return_history:
            history.append(gap_mid)

        if abs(gap_mid) < abs(gap_best):
            prior_best, gap_best = pi_mid.copy(), gap_mid

        if abs(gap_mid) <= eps:
            if return_history:
                return pi_mid, history
            return pi_mid

        # Narrow the bisection interval
        if gap_lower * gap_mid > 0:
            prior_lower, gap_lower = pi_mid, gap_mid
        else:
            prior_upper, gap_upper = pi_mid, gap_mid

        if (prior_upper[0] - prior_lower[0]) <= prior_update_eps:
            break

    if return_history:
        return prior_best, history
    else:
        return prior_best
