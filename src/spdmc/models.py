import inspect

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import dbc
from dbc.discretizers import (
    KMeansDiscretizer,
    KMeansClasswiseDiscretizer,
    CMeansDiscretizer,
    CMeansClasswiseDiscretizer,
    GaussianMixtureDiscretizer,
    GaussianMixtureClasswiseDiscretizer,
    TreeLeafGMMDiscretizer,
)

_DISCRETE_MODEL_REGISTRY = {
    "KMeans": (KMeansDiscretizer, {}),
    "KMeans-ClassWise": (KMeansClasswiseDiscretizer, {}),
    "CMeans": (CMeansDiscretizer, {"use_kmeans": True}),
    "CMeans-ClassWise": (CMeansClasswiseDiscretizer, {"use_kmeans": True}),
    "DiagGMM": (GaussianMixtureDiscretizer, {"covariance_type": "diag"}),
    "DiagGMM-ClassWise": (
        GaussianMixtureClasswiseDiscretizer,
        {"covariance_type": "diag"},
    ),
    "Tree-DiagGMM": (TreeLeafGMMDiscretizer, {"covariance_type": "diag"}),
    # "Tree-FullGMM": (TreeLeafGMMDiscretizer, {"covariance_type": "full"}),
}

_SUPPORTED_MODEL_PREFIXES = {"DMC", "DBC"}


def _filter_params(cls, params: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters) - {"self"}

    unknown = set(params) - accepted
    if unknown:
        raise ValueError(f"Unknown parameters: {unknown}")

    return params


def _split_discrete_model_name(model_name: str) -> tuple[str, str] | None:
    prefix, sep, suffix = model_name.partition("-")
    if not sep or prefix not in _SUPPORTED_MODEL_PREFIXES:
        return None
    return prefix, suffix


def build_model_from_params(
    model_name: str,
    params: dict,
    loss_function="01",
    minmax_eps: float = 1e-3,
    random_seed: int = 0,
):
    """Build a fresh estimator from model name and params dict."""
    discrete_model = _split_discrete_model_name(model_name)
    if discrete_model is not None:
        prefix, suffix = discrete_model
        if suffix not in _DISCRETE_MODEL_REGISTRY:
            raise ValueError(f"Unknown model_name: {model_name}")

        disc_cls, extra_kwargs = _DISCRETE_MODEL_REGISTRY[suffix]
        filtered_params = _filter_params(disc_cls, params)
        disc = disc_cls(**filtered_params, **extra_kwargs, random_state=random_seed)
        return dbc.DiscreteBayesianClassifier(
            discretizer=disc,
            minmax=(prefix == "DMC"),
            minmax_eps=minmax_eps,
            loss_function=loss_function,

        )

    if model_name == "Weighted Logistic Regression":
        return LogisticRegression(
            **params,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed,
        )

    if model_name == "Weighted Decision Tree":
        return DecisionTreeClassifier(
            **params,
            class_weight="balanced",
            random_state=random_seed,
        )

    raise ValueError(f"Unknown model_name: {model_name}")
