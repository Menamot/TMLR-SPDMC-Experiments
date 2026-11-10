"""Normalized class scores for the reviewer-supplement ROC-AUC protocol.

Normalization changes the input scores, never the ROC-AUC definition. DBC/DMC
scores represent normalized nonnegative class support at the prediction prior;
they are not claimed to be calibrated true posteriors. Weighted classifier
baselines retain their native class-probability outputs. SDMC retains its native
action probabilities, whose semantics differ from class-posterior scores.

Saved DBC/DMC negative action risks can recover the class support exactly for
the zero-one and positive off-diagonal binary losses used in these experiments.
Prediction labels must continue to come from the original minimum-risk rule,
especially with asymmetric costs: argmax class support need not minimize cost.
"""

import warnings

import numpy as np


AUC_SCORE_PROTOCOL = "normalized_class_scores_v1"
SCORE_SEMANTICS = {
    "DBC/DMC": "normalized class support at the model prediction prior",
    "weighted_baseline": "native predict_proba class probabilities",
    "SDMC": "native action probabilities; not class posterior probabilities",
}


class ClassSupportRoundoffWarning(UserWarning):
    """Reports the count of tiny negative supports replaced with zero."""


def _score_matrix(values, name):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must have shape (n_samples >= 1, n_classes >= 2).")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def normalize_class_support(class_support, *, negative_tolerance=1e-12):
    """Return row-normalized nonnegative class support without mutating input.

    ``negative_tolerance`` is an absolute tolerance in support units. Values in
    [-negative_tolerance, 0) are treated as numerical roundoff, with their exact
    count and minimum value disclosed in a ``ClassSupportRoundoffWarning``.
    More negative values, nonfinite entries, and zero-mass rows are errors.
    Setting the tolerance to zero disallows all negative entries.
    """
    if not np.isfinite(negative_tolerance) or negative_tolerance < 0:
        raise ValueError("negative_tolerance must be finite and nonnegative.")
    support = _score_matrix(class_support, "class_support")
    minimum = float(support.min())
    if minimum < -negative_tolerance:
        raise ValueError(
            f"Class support contains a negative value ({minimum:.17g}) beyond "
            f"the absolute roundoff tolerance {negative_tolerance:.17g}."
        )
    negative = support < 0
    negative_count = int(np.count_nonzero(negative))
    if negative_count:
        warnings.warn(
            f"Replaced {negative_count} negative class-support entries with zero "
            f"within absolute roundoff tolerance {negative_tolerance:.17g}; "
            f"minimum={minimum:.17g}.",
            ClassSupportRoundoffWarning,
            stacklevel=2,
        )
        support = support.copy()
        support[negative] = 0
    # Scaling first prevents overflow for large but finite nonnegative support.
    scale = support.max(axis=1, keepdims=True)
    if np.any(scale <= 0):
        raise ValueError("Every class-support row must have strictly positive mass.")
    scaled = support / scale
    return scaled / scaled.sum(axis=1, keepdims=True)


def normalized_class_scores_from_negative_risks(scores, loss):
    """Recover and normalize class support from saved negative action risks.

    ``scores`` has shape (n_samples, n_classes). Loss rows index true classes
    and columns index actions, so ``risk = class_support @ loss``. Supported:

    * Zero-one loss: m = sum(risk)/(K-1), class_support = m - risk.
    * Binary [[0, a], [b, 0]], a,b > 0: support = (risk_1/a, risk_0/b).

    All other loss matrices are rejected; there is no generic inverse-risk
    probability conversion. The output columns keep the input class order.
    Tiny negative reconstructed support is handled, and explicitly reported,
    by ``normalize_class_support``.
    """
    negative_risk = _score_matrix(scores, "scores")
    n_classes = negative_risk.shape[1]
    loss = np.asarray(loss, dtype=float)
    if loss.shape != (n_classes, n_classes) or not np.isfinite(loss).all():
        raise ValueError("loss must be a finite square matrix matching score columns.")
    risk = -negative_risk
    zero_one = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    if np.array_equal(loss, zero_one):
        # Dividing before summation reduces avoidable intermediate overflow.
        mass = (risk / (n_classes - 1)).sum(axis=1, keepdims=True)
        support = mass - risk
    elif (
        n_classes == 2
        and loss[0, 0] == 0
        and loss[1, 1] == 0
        and loss[0, 1] > 0
        and loss[1, 0] > 0
    ):
        support = np.column_stack(
            (risk[:, 1] / loss[0, 1], risk[:, 0] / loss[1, 0])
        )
    else:
        raise ValueError("Unsupported loss: require zero-one or positive binary off-diagonal costs.")
    return normalize_class_support(support)
