import numpy as np
from dbc.utils import compute_class_conditional_risk_with_labels


def compute_class_conditional_risk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_function: np.ndarray,
    classes: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-class conditional risk from predictions."""
    return compute_class_conditional_risk_with_labels(
        y_true=y_true,
        y_pred=y_pred,
        loss_function=loss_function,
        classes=classes,
    )


def global_risk(class_risks: np.ndarray, prior: np.ndarray) -> float:
    """Weighted sum of per-class risks: sum(pi_i * R_i)."""
    return float(np.sum(np.asarray(prior) * np.asarray(class_risks)))


def max_class_risk(class_risks: np.ndarray) -> float:
    """Maximum per-class risk."""
    return float(np.max(class_risks))


def risk_range(class_risks: np.ndarray) -> float:
    """Difference between max and min per-class risk."""
    return float(np.max(class_risks) - np.min(class_risks))


def risk_variance(class_risks: np.ndarray) -> float:
    """Variance of per-class risks."""
    return float(np.var(class_risks))


def compute_all_metrics(
    class_risks: np.ndarray, prior: np.ndarray,
) -> dict:
    """Compute all four risk metrics at once."""
    return {
        "global_risk": global_risk(class_risks, prior),
        "max_class_risk": max_class_risk(class_risks),
        "class_risk_range": risk_range(class_risks),
        "class_risk_variance": risk_variance(class_risks),
    }
