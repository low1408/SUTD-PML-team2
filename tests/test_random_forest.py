"""Pure-NumPy random forest against scikit-learn.

Synthetic data only: these tests check the Gini split search, bootstrap and
averaging, not real-cohort performance.

Run: MPLBACKEND=Agg .venv/bin/python -m unittest tests.test_random_forest -v
"""

import ast
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from sepsis_warning import random_forest, histogram_trees
from sepsis_warning.baselines import auroc
from sepsis_warning.random_forest import NumpyRandomForest
from sepsis_warning.logistic import calibrate_frozen, sigmoid


def synthetic(n, seed, features=10):
    """Imbalanced labels driven by an interaction, a step and missingness.
    Includes an empty and a constant column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, features))
    missing = rng.random(X.shape) < 0.2
    score = X[:, 0] * X[:, 1] + 2.0 * (X[:, 2] > 0.5) + 1.5 * missing[:, 3] - 3.5
    y = (rng.random(n) < sigmoid(score)).astype(int)
    X[missing] = np.nan
    X = np.column_stack([X, np.full(n, np.nan), np.full(n, 3.0)])
    columns = [f"f{i}" for i in range(X.shape[1])]
    table = pd.DataFrame(X, columns=columns)
    table["label"] = y
    return table, columns


def integer_data(n, seed):
    """Few distinct values and no missing data, so binning loses nothing."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 12, size=(n, 6)).astype(float)
    score = 0.4 * X[:, 0] - 0.3 * X[:, 1] + (X[:, 2] > 6) - 3.0
    return X, (rng.random(n) < sigmoid(score)).astype(int)


class SingleTree(unittest.TestCase):
    def test_unbagged_tree_equals_sklearn_decision_tree(self):
        X, y = integer_data(5000, 0)
        ours = NumpyRandomForest(n_estimators=1, bootstrap=False, max_features=None, max_depth=5).fit(X, y)
        reference = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=0).fit(X, y)
        self.assertEqual(ours.trees_[0].n_leaves, reference.get_n_leaves())
        np.testing.assert_allclose(ours.predict_proba(X), reference.predict_proba(X), atol=1e-12)

    def test_unweighted_tree_equals_sklearn_decision_tree(self):
        X, y = integer_data(5000, 1)
        ours = NumpyRandomForest(
            n_estimators=1, bootstrap=False, max_features=None, max_depth=4, class_weight=None).fit(X, y)
        reference = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0).fit(X, y)
        np.testing.assert_allclose(ours.predict_proba(X), reference.predict_proba(X), atol=1e-12)

    def test_depth_and_leaf_size_limits(self):
        table, columns = synthetic(3000, 3)
        X = table[columns].to_numpy()
        model = NumpyRandomForest(n_estimators=5, max_depth=3, min_samples_leaf=40, bootstrap=False).fit(
            X, table["label"])
        for tree in model.trees_:
            self.assertLessEqual(tree.n_leaves, 8)
            leaf_sizes = np.bincount(tree.apply(X), minlength=len(tree.left))[tree.left < 0]
            self.assertGreaterEqual(leaf_sizes.min(), 40)


class Forest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.columns = synthetic(8000, 5)
        cls.validation, _ = synthetic(4000, 6)
        cls.model = NumpyRandomForest(seed=0).fit(cls.train[cls.columns], cls.train["label"])
        cls.scores = cls.model.predict_proba(cls.validation[cls.columns])[:, 1]

    def test_ranks_about_as_well_as_sklearn(self):
        # Different random streams, binned splits and no imputation, so only
        # the quality of the ranking is comparable, not individual scores.
        reference = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=80, max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=0),
        ).fit(self.train[self.columns], self.train["label"])
        theirs = reference.predict_proba(self.validation[self.columns])[:, 1]
        label = self.validation["label"].to_numpy()
        self.assertGreater(auroc(label, self.scores), auroc(label, theirs) - 0.02)
        self.assertGreater(np.corrcoef(self.scores, theirs)[0, 1], 0.9)

    def test_probabilities_are_valid(self):
        proba = self.model.predict_proba(self.validation[self.columns])
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        self.assertTrue(np.all(np.isfinite(self.model.decision_function(self.validation[self.columns]))))

    def test_same_seed_same_forest_different_seed_different_forest(self):
        X, y = self.train[self.columns], self.train["label"]
        again = NumpyRandomForest(seed=0).fit(X, y).predict_proba(self.validation[self.columns])[:, 1]
        other = NumpyRandomForest(seed=1).fit(X, y).predict_proba(self.validation[self.columns])[:, 1]
        np.testing.assert_array_equal(again, self.scores)
        self.assertGreater(np.abs(other - self.scores).max(), 1e-3)

    def test_trees_differ_and_averaging_helps(self):
        label = self.validation["label"].to_numpy()
        X = self.validation[self.columns].to_numpy()
        single = [auroc(label, tree.predict(X)) for tree in self.model.trees_]
        self.assertEqual(len(self.model.trees_), 80)
        self.assertGreater(np.std(single), 0)
        self.assertGreater(auroc(label, self.scores), np.mean(single))

    def test_each_node_searches_a_feature_subset(self):
        # With one candidate feature per node, the strongest feature cannot
        # take every root split as it does when all features are searched.
        X, y = self.train[self.columns], self.train["label"]
        narrow = NumpyRandomForest(n_estimators=30, max_features=1, seed=0).fit(X, y)
        full = NumpyRandomForest(n_estimators=30, max_features=None, seed=0).fit(X, y)
        self.assertGreater(len({int(tree.feature[0]) for tree in narrow.trees_}), 3)
        self.assertLessEqual(len({int(tree.feature[0]) for tree in full.trees_}), 2)

    def test_useless_columns_are_never_split_on(self):
        np.testing.assert_array_equal(self.model.split_counts()[-2:], [0, 0])

    def test_platt_calibration_and_joblib_round_trip(self):
        calibration, _ = synthetic(4000, 7)
        calibrated = calibrate_frozen(self.model, calibration, self.columns)
        label = self.validation["label"]
        probability = calibrated.predict_proba(self.validation[self.columns])[:, 1]
        self.assertAlmostEqual(probability.mean(), label.mean(), delta=0.02)
        self.assertLess(np.mean((probability - label) ** 2), np.mean((self.scores - label) ** 2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(calibrated, path)
            loaded = joblib.load(path)
        np.testing.assert_array_equal(loaded.predict_proba(self.validation[self.columns])[:, 1], probability)

    def test_rejects_bad_input(self):
        X = self.train[self.columns]
        with self.assertRaises(ValueError):
            self.model.predict_proba(X.iloc[:, :3])
        with self.assertRaises(ValueError):
            NumpyRandomForest().fit(X, np.ones(len(X)))
        with self.assertRaises(ValueError):
            NumpyRandomForest(max_features="log2")


class NoSklearn(unittest.TestCase):
    def test_modules_import_only_numpy_and_local_code(self):
        for module in (random_forest, histogram_trees):
            imported = set()
            for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.add((node.module or "").split(".")[0])
            self.assertLessEqual(imported, {"numpy", "logistic", "histogram_trees"})


if __name__ == "__main__":
    unittest.main()
