"""Binned decision-tree builder shared by the NumPy boosting and forest models.

No scikit-learn is imported here.

1. ``FeatureBinner`` maps each feature once to at most 255 integer bins, plus
   one bin reserved for missing values, so no imputation is needed.
2. ``grow_tree`` finds splits from per-bin sums (``np.bincount``) instead of
   sorting values at every node. At each split the missing-value bin is tried
   on both sides and sent wherever the gain is larger.

Every row carries a "gradient" ``g`` and a "Hessian" ``h``. A split's gain is

    GL^2 / (HL + l2) + GR^2 / (HR + l2) - G^2 / (H + l2)

where capitals are sums over a node. Gradient boosting passes the log-loss
derivatives. With ``g = weight * y``, ``h = weight`` and ``l2 = 0`` the same
expression is half the weighted Gini impurity decrease, which is what the
random forest uses.
"""

import numpy as np

from .logistic import _as_matrix


class FeatureBinner:
    """Map each feature to ``uint8`` bins; the last bin holds missing values.

    Thresholds are learned from the rows passed to ``fit`` only: midpoints
    between distinct values when there are few of them, otherwise quantiles.
    Value ``x`` falls in bin ``b`` when ``thresholds[b-1] < x <= thresholds[b]``.
    """

    def __init__(self, max_bins=255, subsample=200_000, seed=0):
        if not 2 <= max_bins <= 255:
            raise ValueError("max_bins must be between 2 and 255")
        self.max_bins = max_bins
        self.subsample = subsample
        self.seed = seed

    @property
    def missing_bin(self):
        return self.max_bins

    def fit(self, X):
        X = _as_matrix(X)
        if len(X) > self.subsample:
            rows = np.random.default_rng(self.seed).choice(len(X), self.subsample, replace=False)
            X = X[np.sort(rows)]
        percentiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]
        self.thresholds_ = []
        for feature in range(X.shape[1]):
            column = X[:, feature]
            distinct = np.unique(column[~np.isnan(column)])
            if len(distinct) <= self.max_bins:
                thresholds = (distinct[:-1] + distinct[1:]) * 0.5
            else:
                present = column[~np.isnan(column)]
                thresholds = np.unique(np.percentile(present, percentiles, method="midpoint"))
            self.thresholds_.append(thresholds)
        return self

    def transform(self, X):
        X = _as_matrix(X)
        if X.shape[1] != len(self.thresholds_):
            raise ValueError("X has a different number of features than the fitted binner")
        # Column-major so each feature's codes are contiguous for bincount.
        binned = np.empty(X.shape, dtype=np.uint8, order="F")
        for feature, thresholds in enumerate(self.thresholds_):
            column = X[:, feature]
            codes = np.searchsorted(thresholds, column, side="left")
            codes[np.isnan(column)] = self.missing_bin
            binned[:, feature] = codes
        return binned


def splittable_features(binned):
    """Features with more than one bin in training; the rest can never split."""
    return np.array([f for f in range(binned.shape[1]) if binned[:, f].min() != binned[:, f].max()], dtype=np.intp)


class Tree:
    """A fitted tree stored as flat arrays. ``left < 0`` marks a leaf."""

    def __init__(self, feature, threshold, missing_left, left, right):
        self.feature = np.asarray(feature, dtype=np.intp)
        self.threshold = np.asarray(threshold, dtype=np.float64)
        self.missing_left = np.asarray(missing_left, dtype=bool)
        self.left = np.asarray(left, dtype=np.intp)
        self.right = np.asarray(right, dtype=np.intp)
        self.value = np.zeros(len(self.feature))

    @property
    def n_leaves(self):
        return int((self.left < 0).sum())

    def apply(self, X):
        """Index of the leaf each row of raw (unbinned) ``X`` lands in."""
        node = np.zeros(len(X), dtype=np.intp)
        rows = np.flatnonzero(self.left[node] >= 0)
        while rows.size:
            at = node[rows]
            x = X[rows, self.feature[at]]
            go_left = np.where(np.isnan(x), self.missing_left[at], x <= self.threshold[at])
            node[rows] = np.where(go_left, self.left[at], self.right[at])
            rows = rows[self.left[node[rows]] >= 0]
        return node

    def predict(self, X):
        return self.value[self.apply(X)]


def _histograms(binned, gradient, hessian, rows, features, width):
    """Per-bin sums of gradient, Hessian and row count for the given features."""
    histogram = np.empty((3, len(features), width))
    if rows is not None:
        gradient, hessian = gradient[rows], hessian[rows]
    for slot, feature in enumerate(features):
        codes = binned[:, feature] if rows is None else binned[rows, feature]
        histogram[0, slot] = np.bincount(codes, weights=gradient, minlength=width)
        histogram[1, slot] = np.bincount(codes, weights=hessian, minlength=width)
        histogram[2, slot] = np.bincount(codes, minlength=width)
    return histogram


def _best_split(histogram, total, min_samples_leaf, l2, min_hessian):
    """Best (gain, slot, bin, missing_left) for a node, or None.

    Rows with ``bin <= b`` go left. Each candidate is scored twice: with the
    missing-value bin sent right, and with it sent left.
    """
    G, H, C = total
    present, missing = histogram[:, :, :-1], histogram[:, :, -1:]
    left = np.cumsum(present, axis=2)                      # missing goes right
    left = np.stack([left, left + missing], axis=1)        # [stat, case, slot, bin]
    GL, HL, CL = left
    GR, HR, CR = G - GL, H - HL, C - CL
    valid = (CL >= min_samples_leaf) & (CR >= min_samples_leaf) & (HL >= min_hessian) & (HR >= min_hessian)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = GL * GL / (HL + l2) + GR * GR / (HR + l2) - G * G / (H + l2)
    gain = np.where(valid, gain, -np.inf)
    case, slot, bin_index = np.unravel_index(np.argmax(gain), gain.shape)
    best = gain[case, slot, bin_index]
    if not best > 1e-12 * max(1.0, abs(G * G / (H + l2))):
        return None
    return float(best), int(slot), int(bin_index), bool(case)


def grow_tree(binned, gradient, hessian, thresholds, features, *, rows=None, max_leaf_nodes=None,
              max_depth=None, min_samples_leaf=20, l2=0.0, min_hessian=0.0,
              split_features=None, rng=None, missing_bin=255):
    """Grow one tree best-first and return ``(tree, leaves)``.

    ``leaves`` is a list of ``(node, rows, G, H)``; the caller turns the sums
    into leaf values and writes them to ``tree.value``. ``rows`` restricts the
    tree to a subset of training rows (``None`` means all of them).

    With ``split_features=None`` every feature in ``features`` is searched at
    every node, and a child's histogram is obtained as parent minus sibling.
    With ``split_features=k`` each node searches a fresh random ``k`` of them
    (drawn from ``rng``), as a random forest does.
    """
    width = missing_bin + 1  # the binner puts missing values in its last bin
    features = np.asarray(features, dtype=np.intp)
    sample = split_features is not None and split_features < len(features)
    feature_of, threshold_of, missing_left_of, left_of, right_of = [-1], [np.nan], [False], [-1], [-1]

    def node_total(node_rows):
        if node_rows is None:
            return gradient.sum(), hessian.sum(), float(len(gradient))
        return gradient[node_rows].sum(), hessian[node_rows].sum(), float(len(node_rows))

    def make_leaf(node, node_rows, histogram, depth):
        total = node_total(node_rows)
        split = candidates = None
        can_split = (max_depth is None or depth < max_depth) and total[2] >= 2 * min_samples_leaf
        if can_split and len(features):
            if sample:
                candidates = rng.choice(features, split_features, replace=False)
                histogram = _histograms(binned, gradient, hessian, node_rows, candidates, width)
            else:
                candidates = features
                if histogram is None:
                    histogram = _histograms(binned, gradient, hessian, node_rows, candidates, width)
            split = _best_split(histogram, total, min_samples_leaf, l2, min_hessian)
        return {
            "node": node, "rows": node_rows, "total": total, "depth": depth, "split": split,
            "candidates": candidates, "histogram": None if sample or split is None else histogram,
        }

    leaves = [make_leaf(0, rows, None, 0)]
    open_leaves = [leaf for leaf in leaves if leaf["split"] is not None]

    while open_leaves and (max_leaf_nodes is None or len(leaves) < max_leaf_nodes):
        parent = max(open_leaves, key=lambda leaf: leaf["split"][0])
        open_leaves.remove(parent)
        leaves.remove(parent)
        _, slot, bin_index, missing_left = parent["split"]
        feature = int(parent["candidates"][slot])

        node_rows = parent["rows"]
        codes = binned[:, feature] if node_rows is None else binned[node_rows, feature]
        go_left = codes <= bin_index
        if missing_left:
            go_left |= codes == width - 1
        if node_rows is None:
            node_rows = np.arange(len(codes))
        sides = [node_rows[go_left], node_rows[~go_left]]

        histograms = [None, None]
        if not sample:
            # Build the smaller child's histogram; the sibling is parent minus it.
            small = 0 if len(sides[0]) <= len(sides[1]) else 1
            histograms[small] = _histograms(binned, gradient, hessian, sides[small], features, width)
            histograms[1 - small] = parent["histogram"] - histograms[small]
        parent["histogram"] = None

        children = []
        for side in (0, 1):
            node = len(feature_of)
            for store, default in ((feature_of, -1), (threshold_of, np.nan), (missing_left_of, False),
                                   (left_of, -1), (right_of, -1)):
                store.append(default)
            children.append(node)
            leaf = make_leaf(node, sides[side], histograms[side], parent["depth"] + 1)
            leaves.append(leaf)
            if leaf["split"] is not None:
                open_leaves.append(leaf)

        cuts = thresholds[feature]
        node = parent["node"]
        feature_of[node] = feature
        # Past the last threshold every observed value goes left.
        threshold_of[node] = cuts[bin_index] if bin_index < len(cuts) else np.inf
        missing_left_of[node] = missing_left
        left_of[node], right_of[node] = children

    tree = Tree(feature_of, threshold_of, missing_left_of, left_of, right_of)
    return tree, [(leaf["node"], leaf["rows"], leaf["total"][0], leaf["total"][1]) for leaf in leaves]
