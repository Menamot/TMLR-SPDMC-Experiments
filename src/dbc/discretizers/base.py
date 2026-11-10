import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class BaseDiscretizer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for all discretizers.

    Subclasses must implement ``fit`` and ``transform`` and set the class-level
    ``_is_soft`` flag to indicate whether the output is a soft (probabilistic)
    or hard (one-hot) membership matrix.

    Notes:
        - Inherits from ``BaseEstimator`` for ``get_params`` / ``set_params``
          and ``GridSearchCV`` compatibility.
        - Inherits from ``TransformerMixin`` to provide ``fit_transform`` for free.
        - Learned attributes must use a trailing underscore (e.g., ``model_``).
        - ``transform`` must return an ndarray of shape ``(n_clusters, n_samples)``.

    Attributes:
        _is_soft: Class-level boolean. ``True`` for soft (fuzzy/probabilistic)
            discretizers, ``False`` for hard (one-hot) discretizers.
    """

    _is_soft: bool

    @abstractmethod
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "BaseDiscretizer":
        """Fits the discretizer on data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Optional class labels of shape (n_samples,). Required by
                class-wise discretizers; ignored by unsupervised ones.

        Returns:
            self
        """
        return self

    @abstractmethod
    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transforms data into a cluster membership matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Membership matrix of shape (n_clusters, n_samples). For hard
            discretizers each column is a one-hot vector; for soft
            discretizers each column sums to 1.
        """
        pass
