"""Histogram gradient-boosted trees in pure NumPy (K03).

No scikit-learn is imported here. The design follows LightGBM and
scikit-learn's ``HistGradientBoostingClassifier``, which stays in
``modeling.py`` as the reference benchmark:

1. Each feature is binned once into at most 255 integer bins, plus one bin
   reserved for missing values, so no imputation is needed.
2. Each tree is fitted to the gradients and Hessians of the weighted log-loss.
   A node's split search needs only per-bin sums (``np.bincount``), not a sort.
3. Trees grow leaf-wise: the leaf with the largest gain is split next, until
   ``max_leaf_nodes`` is reached. At every split the missing-value bin is
   tried on both sides and sent wherever the gain is larger.
"""

import numpy as np

from .logistic import _as_matrix, _binary_labels, _log_loss_terms, row_weights, sigmoid
from .histogram_trees import FeatureBinner, grow_tree, splittable_features

MIN_HESSIAN = 1e-3  # a child needs at least this much Hessian mass


class NumpyGradientBoosting:
    """Binary classifier: boosted histogram trees on the weighted log-loss.

    Parameters mirror ``HistGradientBoostingClassifier``. There is no early
    stopping and no internal validation split, so no patient-hour rows are
    held out behind the caller's back. The fit is deterministic; ``seed`` only
    chooses the rows used to learn bin thresholds when there are more than
    200,000.
    """

    def __init__(self, learning_rate=0.06, max_iter=100, max_leaf_nodes=15, max_depth=None,
                 min_samples_leaf=20, l2_regularization=2.0, max_bins=255,
                 class_weight="balanced", seed=0):
        if class_weight not in ("balanced", None):
            raise ValueError("class_weight must be 'balanced' or None")
        if max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes must be at least 2")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.class_weight = class_weight
        self.seed = seed

    # ---- boosting ---------------------------------------------------------

    def fit(self, X, y, sample_weight=None):
        """Fit on rows ``X`` with labels ``y``.

        ``sample_weight`` is an optional non-negative weight per row. It
        multiplies the class weights, as in scikit-learn. ``min_samples_leaf``
        still counts rows, whatever their weight.
        """
        X = _as_matrix(X)
        y = _binary_labels(y)
        if len(y) != len(X):
            raise ValueError("X and y have different numbers of rows")
        n = len(y)
        weight = row_weights(y, self.class_weight, sample_weight)

        self.binner_ = FeatureBinner(self.max_bins, seed=self.seed).fit(X)
        binned = self.binner_.transform(X)
        thresholds = self.binner_.thresholds_
        features = splittable_features(binned)

        prior = np.clip((weight @ y) / weight.sum(), 1e-12, 1 - 1e-12)
        self.baseline_ = float(np.log(prior / (1.0 - prior)))
        raw = np.full(n, self.baseline_)
        self.trees_ = []
        self.loss_history_ = [float(weight @ _log_loss_terms(raw, y) / weight.sum())]

        for _ in range(self.max_iter):
            probability = sigmoid(raw)
            gradient = weight * (probability - y)
            hessian = weight * probability * (1.0 - probability)
            tree, leaves = grow_tree(
                binned, gradient, hessian, thresholds, features,
                max_leaf_nodes=self.max_leaf_nodes, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, l2=self.l2_regularization, min_hessian=MIN_HESSIAN,
                missing_bin=self.binner_.missing_bin,
            )
            if tree.n_leaves == 1:
                break  # no split improves the loss any further
            for node, rows, G, H in leaves:
                tree.value[node] = -self.learning_rate * G / (H + self.l2_regularization + 1e-15)
                raw[rows] += tree.value[node]
            self.trees_.append(tree)
            self.loss_history_.append(float(weight @ _log_loss_terms(raw, y) / weight.sum()))

        self.n_iter_ = len(self.trees_)
        self.n_features_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_:
            raise ValueError("X has a different number of features than the fitted model")
        raw = np.full(len(X), self.baseline_)
        for tree in self.trees_:
            raw += tree.predict(X)
        return raw

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def split_counts(self):
        """How many splits use each feature. Describes the model, not causes."""
        counts = np.zeros(self.n_features_, dtype=int)
        for tree in self.trees_:
            np.add.at(counts, tree.feature[tree.left >= 0], 1)
        return counts
