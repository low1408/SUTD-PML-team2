"""Pure-NumPy histogram gradient boosting against scikit-learn.

Synthetic data only: these tests check the binning, split search and boosting
maths, not real-cohort performance.

Run: MPLBACKEND=Agg .venv/bin/python -m unittest tests.test_gradient_boosting -v
"""

import ast
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from sepsis_warning import gradient_boosting
from sepsis_warning.baselines import auroc
from sepsis_warning.gradient_boosting import NumpyGradientBoosting
from sepsis_warning.histogram_trees import FeatureBinner
from sepsis_warning.logistic import calibrate_frozen, sigmoid


def synthetic(n, seed, features=10):
    """Imbalanced labels driven by an interaction, a step and missingness, so
    a linear model cannot fit them. Includes an empty and a constant column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, features))
    X[:, 4] = np.round(X[:, 4])  # a feature with few distinct values
    missing = rng.random(X.shape) < 0.2
    score = X[:, 0] * X[:, 1] + 2.0 * (X[:, 2] > 0.5) + 1.5 * missing[:, 3] - 3.5
    y = (rng.random(n) < sigmoid(score)).astype(int)
    X[missing] = np.nan
    X = np.column_stack([X, np.full(n, np.nan), np.full(n, 3.0)])
    columns = [f"f{i}" for i in range(X.shape[1])]
    table = pd.DataFrame(X, columns=columns)
    table["label"] = y
    return table, columns


def sklearn_reference(max_iter=100):
    return HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=max_iter, max_leaf_nodes=15, l2_regularization=2,
        class_weight="balanced", early_stopping=False, random_state=0,
    )


class Binning(unittest.TestCase):
    def test_few_distinct_values_get_midpoint_thresholds(self):
        X = np.array([[1.0], [1.0], [2.0], [4.0], [np.nan]])
        binner = FeatureBinner().fit(X)
        np.testing.assert_array_equal(binner.thresholds_[0], [1.5, 3.0])
        np.testing.assert_array_equal(binner.transform(X)[:, 0], [0, 0, 1, 2, 255])

    def test_unseen_values_and_missing_are_binned_from_training_thresholds(self):
        binner = FeatureBinner().fit(np.array([[1.0], [2.0], [4.0]]))
        codes = binner.transform(np.array([[-100.0], [1.5], [1.6], [100.0], [np.nan]]))[:, 0]
        np.testing.assert_array_equal(codes, [0, 0, 1, 2, 255])

    def test_many_distinct_values_use_at_most_max_bins(self):
        X = np.random.default_rng(0).normal(size=(5000, 1))
        binner = FeatureBinner(max_bins=32).fit(X)
        codes = binner.transform(X)[:, 0]
        self.assertLessEqual(codes.max(), 31)
        counts = np.bincount(codes, minlength=32)
        self.assertLess(counts.max() - counts.min(), 5)  # quantile bins are near-equal


class TreeBehaviour(unittest.TestCase):
    def test_recovers_a_single_step(self):
        x = np.linspace(-1, 1, 400)
        y = (x > 0.3).astype(int)
        model = NumpyGradientBoosting(max_iter=1, learning_rate=1.0, l2_regularization=0.0).fit(x[:, None], y)
        tree = model.trees_[0]
        self.assertEqual(tree.feature[0], 0)
        self.assertAlmostEqual(tree.threshold[0], 0.3, delta=0.01)
        raw = model.decision_function(x[:, None])
        self.assertTrue(np.all(raw[y == 1] > 0) and np.all(raw[y == 0] < 0))

    def test_missingness_alone_can_be_split_on(self):
        rng = np.random.default_rng(1)
        y = (rng.random(600) < 0.5).astype(int)
        x = np.where(y == 1, np.nan, 5.0)  # one observed value, missing iff positive
        model = NumpyGradientBoosting(max_iter=5).fit(x[:, None], y)
        p = model.predict_proba(np.array([[np.nan], [5.0], [-9.0]]))[:, 1]
        self.assertGreater(p[0], 0.5)
        self.assertLess(p[1], 0.5)
        self.assertEqual(p[1], p[2])  # every observed value takes the same branch

    def test_leaf_limits_are_respected(self):
        table, columns = synthetic(3000, 3)
        model = NumpyGradientBoosting(max_iter=10, max_leaf_nodes=6, min_samples_leaf=50).fit(
            table[columns], table["label"])
        X = table[columns].to_numpy()
        for tree in model.trees_:
            self.assertLessEqual(tree.n_leaves, 6)
            node = np.zeros(len(X), dtype=int)
            for _ in range(8):  # route training rows to their leaves
                inner = tree.left[node] >= 0
                x = X[np.arange(len(X)), np.where(inner, tree.feature[node], 0)]
                left = np.where(np.isnan(x), tree.missing_left[node], x <= tree.threshold[node])
                node = np.where(inner, np.where(left, tree.left[node], tree.right[node]), node)
            self.assertGreaterEqual(np.bincount(node)[np.unique(node)].min(), 50)

    def test_max_depth_one_gives_stumps(self):
        table, columns = synthetic(2000, 4)
        model = NumpyGradientBoosting(max_iter=5, max_depth=1).fit(table[columns], table["label"])
        self.assertTrue(all(tree.n_leaves == 2 for tree in model.trees_))


class AgainstSklearn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.columns = synthetic(8000, 5)
        cls.validation, _ = synthetic(4000, 6)
        cls.model = NumpyGradientBoosting().fit(cls.train[cls.columns], cls.train["label"])

    def test_first_tree_is_the_same_tree(self):
        ours = NumpyGradientBoosting(max_iter=1).fit(self.train[self.columns], self.train["label"])
        reference = sklearn_reference(max_iter=1).fit(self.train[self.columns], self.train["label"])
        np.testing.assert_allclose(
            ours.decision_function(self.validation[self.columns]),
            reference.decision_function(self.validation[self.columns]), atol=1e-6,
        )

    def test_full_model_tracks_reference(self):
        # scikit-learn stores gradients in float32, so later trees can break
        # near-ties differently; scores agree closely but not exactly.
        reference = sklearn_reference().fit(self.train[self.columns], self.train["label"])
        ours = self.model.predict_proba(self.validation[self.columns])[:, 1]
        theirs = reference.predict_proba(self.validation[self.columns])[:, 1]
        label = self.validation["label"].to_numpy()
        self.assertGreater(np.corrcoef(ours, theirs)[0, 1], 0.995)
        self.assertLess(np.abs(ours - theirs).mean(), 0.01)
        self.assertAlmostEqual(auroc(label, ours), auroc(label, theirs), delta=0.01)

    def test_training_loss_falls_every_iteration(self):
        self.assertEqual(self.model.n_iter_, 100)
        self.assertTrue(np.all(np.diff(self.model.loss_history_) < 0))

    def test_training_scores_equal_prediction_on_raw_values(self):
        # Splits found on bins must route raw values the same way.
        X, y = self.train[self.columns], self.train["label"].to_numpy()
        n = len(y)
        weight = np.where(y == 1, n / (2.0 * y.sum()), n / (2.0 * (n - y.sum())))
        raw = self.model.decision_function(X)
        loss = weight @ (np.logaddexp(0.0, raw) - y * raw) / weight.sum()
        self.assertAlmostEqual(loss, self.model.loss_history_[-1], places=10)

    def test_fit_is_deterministic_and_ignores_useless_columns(self):
        again = NumpyGradientBoosting().fit(self.train[self.columns], self.train["label"])
        np.testing.assert_array_equal(
            again.decision_function(self.validation[self.columns]),
            self.model.decision_function(self.validation[self.columns]),
        )
        np.testing.assert_array_equal(self.model.split_counts()[-2:], [0, 0])

    def test_platt_calibration_and_joblib_round_trip(self):
        calibration, _ = synthetic(4000, 7)
        calibrated = calibrate_frozen(self.model, calibration, self.columns)
        label = self.validation["label"]
        raw = self.model.predict_proba(self.validation[self.columns])[:, 1]
        probability = calibrated.predict_proba(self.validation[self.columns])[:, 1]
        self.assertAlmostEqual(probability.mean(), label.mean(), delta=0.02)
        self.assertLess(np.mean((probability - label) ** 2), np.mean((raw - label) ** 2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(calibrated, path)
            loaded = joblib.load(path)
        np.testing.assert_array_equal(loaded.predict_proba(self.validation[self.columns])[:, 1], probability)

    def test_rejects_wrong_feature_count_and_bad_labels(self):
        X = self.train[self.columns]
        with self.assertRaises(ValueError):
            self.model.predict_proba(X.iloc[:, :3])
        with self.assertRaises(ValueError):
            NumpyGradientBoosting().fit(X, np.zeros(len(X)))


class NoSklearn(unittest.TestCase):
    def test_module_imports_only_numpy_and_local_code(self):
        tree = ast.parse(Path(gradient_boosting.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"numpy", "logistic", "histogram_trees"})


if __name__ == "__main__":
    unittest.main()
