import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted, validate_data

from dbc.discretizers.base import BaseDiscretizer
from dbc.utils import _iter_batches


class DecisionTreeDiscretizer(BaseDiscretizer):
    """Hard discretizer based on decision tree leaves.

    Each leaf node is treated as a discrete profile. transform() returns a
    one-hot membership matrix of shape (n_leaves, n_samples).

    Attributes:
        model_: Fitted DecisionTreeClassifier, set after fit.
        leaf_nodes_: Array of leaf node IDs, set after fit.
        _leaf_lookup_: Array of shape (max_node_id + 1,) mapping node IDs to
            leaf indices for O(1) numpy fancy indexing, set after fit.
    """

    _is_soft = False

    def __init__(
        self,
        max_depth: int = 5,
        random_state=None,
        batch_size: int = 16384,
    ):
        """Initializes DecisionTreeDiscretizer.

        Args:
            max_depth: Maximum depth of the decision tree.
            random_state: Random seed for reproducibility.
            batch_size: Number of samples per chunk when applying the tree in
                transform(). Larger values may be faster but use more memory.
        """
        self.max_depth = max_depth
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X, y):
        """Fits a decision tree and extracts leaf node information.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            self
        """
        X = validate_data(self, X, dtype=np.float64)

        self.model_ = DecisionTreeClassifier(
            max_depth=self.max_depth, random_state=self.random_state
        )
        self.model_.fit(X, y)
        # Leaf nodes have tree_.feature == -2 (sklearn TREE_LEAF sentinel)
        self.leaf_nodes_ = np.where(self.model_.tree_.feature == -2)[0]
        # Build a flat lookup array: node_id -> leaf index (faster than a dict in transform)
        self._leaf_lookup_ = np.full(
            int(self.leaf_nodes_.max()) + 1, fill_value=-1, dtype=np.intp
        )
        self._leaf_lookup_[self.leaf_nodes_] = np.arange(len(self.leaf_nodes_), dtype=np.intp)
        return self

    def _apply_batched(self, X):
        n_samples = X.shape[0]
        leaf_assignments = np.empty(n_samples, dtype=int)
        for batch_start, batch_end in _iter_batches(n_samples, self.batch_size):
            leaf_assignments[batch_start:batch_end] = self.model_.apply(
                X[batch_start:batch_end]
            )
        return leaf_assignments

    def transform(self, X):
        """Returns a one-hot leaf membership matrix for each sample.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Membership matrix of shape (n_leaves, n_samples).
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        leaf_node_ids = self._apply_batched(X)
        n_samples = X.shape[0]
        n_leaves = len(self.leaf_nodes_)
        membership_degree = np.zeros((n_leaves, n_samples), dtype=np.float64)
        leaf_indices = self._leaf_lookup_[leaf_node_ids]
        membership_degree[leaf_indices, np.arange(n_samples)] = 1.0
        return membership_degree
