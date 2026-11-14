"""Random forest in pure NumPy (K02), built on the shared binned tree builder.

No scikit-learn is imported here; ``RandomForestClassifier`` stays in
``modeling.py`` as the reference benchmark. Each tree is grown on a bootstrap
sample of the training rows, searches a fresh random subset of features at
every node, and splits on weighted Gini impurity. The forest's probability is
the average of the trees' leaf class fractions.

Two deliberate differences from the reference pipeline:

* Splits are searched over at most 255 quantile bins per feature rather than
  every distinct value.
* Missing values are not imputed. They have their own bin and each split sends
  them to whichever side reduces impurity more.
"""

import numpy as np

from .logistic import _as_matrix, _binary_labels, row_weights
from .histogram_trees import FeatureBinner, grow_tree, splittable_features


class NumpyRandomForest:
    """Binary random-forest classifier.

    Parameters
    ----------
    n_estimators, max_depth, min_samples_leaf : as in scikit-learn.
        ``min_samples_leaf`` counts distinct training rows in a leaf.
    max_features : ``"sqrt"``, an integer, or ``None`` for all features.
    bootstrap : draw ``n`` rows with replacement for each tree.
    class_weight : ``"balanced"`` (weights ``n / (2 * n_class)`` from the full
        training labels) or ``None``.
    seed : controls the bootstrap samples and the per-node feature subsets.
    """

    def __init__(self, n_estimators=80, max_depth=8, min_samples_leaf=20, max_features="sqrt",
                 bootstrap=True, class_weight="balanced", max_bins=255, seed=0):
        if class_weight not in ("balanced", None):
            raise ValueError("class_weight must be 'balanced' or None")
        if not (max_features in ("sqrt", None) or (isinstance(max_features, int) and max_features >= 1)):
            raise ValueError("max_features must be 'sqrt', None or a positive integer")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.max_bins = max_bins
        self.seed = seed

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
        features = splittable_features(binned)
        if self.max_features is None:
            split_features = None
        elif self.max_features == "sqrt":
            split_features = max(1, int(np.sqrt(X.shape[1])))
        else:
            split_features = self.max_features
        if split_features is not None:
            split_features = min(split_features, max(len(features), 1))

        rng = np.random.default_rng(self.seed)
        self.prior_ = float((weight @ y) / weight.sum())
        self.trees_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                draws = np.bincount(rng.integers(n, size=n), minlength=n)
                rows = np.flatnonzero(draws)
                tree_weight = weight * draws
            else:
                rows, tree_weight = None, weight
            # g = w*y and h = w make the builder's gain the weighted Gini decrease.
            tree, leaves = grow_tree(
                binned, tree_weight * y, tree_weight, self.binner_.thresholds_, features,
                rows=rows, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                split_features=split_features, rng=rng, missing_bin=self.binner_.missing_bin,
            )
            for node, _, positive_weight, total_weight in leaves:
                tree.value[node] = positive_weight / total_weight if total_weight > 0 else self.prior_
            self.trees_.append(tree)

        self.n_features_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        return self

    def _positive_probability(self, X):
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_:
            raise ValueError("X has a different number of features than the fitted model")
        total = np.zeros(len(X))
        for tree in self.trees_:
            total += tree.predict(X)
        return total / len(self.trees_)

    def predict_proba(self, X):
        p = self._positive_probability(X)
        return np.column_stack([1.0 - p, p])

    def decision_function(self, X):
        """Log-odds of the averaged probability, the input to Platt scaling.

        Clipped so a unanimous forest gives a finite score.
        """
        p = np.clip(self._positive_probability(X), 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    def split_counts(self):
        """How many splits use each feature. Describes the model, not causes."""
        counts = np.zeros(self.n_features_, dtype=int)
        for tree in self.trees_:
            np.add.at(counts, tree.feature[tree.left >= 0], 1)
        return counts
