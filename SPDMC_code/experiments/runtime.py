"""Cache fixed profile-to-class features during prior optimization."""

from contextlib import contextmanager
import numpy as np
import dbc.utils as dbc_utils


@contextmanager
def cached_training_risks():
    original = dbc_utils.compute_class_conditional_risk
    cache = {}

    def calculate(y, loss_function, p_hat, membership, prior_train, prior_pred):
        key = (id(p_hat), id(membership), id(prior_train))
        if key not in cache:
            pz = dbc_utils.compute_pz(p_hat, prior_train)
            cache.clear()
            cache[key] = (p_hat / pz[None, :]) @ membership
        risk = (cache[key].T * np.asarray(prior_pred).ravel()) @ loss_function
        smallest = np.partition(risk, 1, axis=1)[:, :2]
        margin = smallest[:, 1] - smallest[:, 0]
        if (not np.isfinite(risk).all()
                or np.any(margin <= 1e-11 * (1 + np.max(np.abs(risk), axis=1)))):
            return original(y, loss_function, p_hat, membership, prior_train, prior_pred)
        pred = risk.argmin(axis=1)
        k = len(loss_function)
        counts = np.bincount(y, minlength=k)
        sums = np.bincount(y, weights=loss_function[y, pred], minlength=k)
        return np.divide(sums, counts, out=np.zeros(k), where=counts > 0)

    dbc_utils.compute_class_conditional_risk = calculate
    try:
        yield
    finally:
        dbc_utils.compute_class_conditional_risk = original


def prediction_features(model, X):
    """Diagonal multiplication evaluated elementwise, in batches."""
    pz = dbc_utils.compute_pz(model.p_hat_, model.prior_)
    inv = 1. / pz
    result = np.empty((len(X), len(model.prior_)), dtype=float)
    for start in range(0, len(X), 8192):
        stop = min(start + 8192, len(X))
        membership = model.discretizer.transform(X[start:stop])
        membership *= inv[:, None]
        result[start:stop] = (model.p_hat_ @ membership).T
    return result
