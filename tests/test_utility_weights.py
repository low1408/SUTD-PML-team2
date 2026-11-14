"""Utility-weighted training targets and ``sample_weight`` in the NumPy models.

Synthetic data only: these tests check that the weights encode the published
utility exactly and that the models honour them, not real-cohort performance.

Run: MPLBACKEND=Agg .venv/bin/python -m unittest tests.test_utility_weights -v
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sepsis_warning.random_forest import NumpyRandomForest
from sepsis_warning.gradient_boosting import NumpyGradientBoosting
from sepsis_warning.logistic import NumpyLogisticRegression, sigmoid
from sepsis_warning.policy import utility_components, utility_training_targets


def stay(patient_id, hours, label_from=None, first_hour=1):
    label = np.zeros(hours, dtype=int)
    if label_from is not None:
        label[label_from:] = 1
    return pd.DataFrame({
        "patient_id": patient_id, "hour": np.arange(first_hour, first_hour + hours), "label": label,
    })


def cohort():
    return pd.concat([
        stay("A:never", 40),
        stay("A:late", 60, label_from=30),
        stay("A:early", 25, label_from=2),
        stay("A:at_admission", 12, label_from=0),
        stay("A:short_tail", 20, label_from=18, first_hour=5),
    ], ignore_index=True)


class UtilityTargets(unittest.TestCase):
    def test_pattern_around_onset(self):
        # Label turns on at row 14, so clinical onset is row 20.
        target, weight = utility_training_targets(stay("p", 30, label_from=14), normalize=False)
        np.testing.assert_array_equal(target, [0] * 9 + [1] * 15 + [0] * 6)
        np.testing.assert_allclose(weight[:8], 0.05)       # far too early: small false-alarm cost
        self.assertEqual(weight[8], 0.0)                    # 12 h before onset: utility is indifferent
        np.testing.assert_allclose(weight[9:14], np.arange(1, 6) / 6)  # rewarded before the label starts
        np.testing.assert_allclose(weight[14], 1.0)         # 6 h before onset
        np.testing.assert_allclose(weight[23], 2.0)         # 3 h after onset: most costly miss
        np.testing.assert_array_equal(weight[24:], 0.0)     # utility stops accumulating

    def test_non_septic_stay_is_all_cheap_negatives(self):
        target, weight = utility_training_targets(stay("p", 15), normalize=False)
        np.testing.assert_array_equal(target, 0)
        np.testing.assert_allclose(weight, 0.05)

    def test_weighted_error_equals_utility_regret(self):
        table = cohort()
        target, weight = utility_training_targets(table, normalize=False)
        rng = np.random.default_rng(0)
        for _ in range(20):
            decisions = rng.random(len(table)) < rng.random()
            achieved = best = 0.0
            for _, patient in table.assign(decision=decisions).groupby("patient_id", sort=False):
                positive, negative = utility_components(patient["label"])
                achieved += np.where(patient["decision"], positive, negative).sum()
                best += np.maximum(positive, negative).sum()
            self.assertAlmostEqual((weight * (decisions != target)).sum(), best - achieved, places=10)

    def test_row_order_does_not_matter(self):
        table = cohort()
        target, weight = utility_training_targets(table)
        order = np.random.default_rng(1).permutation(len(table))
        shuffled_target, shuffled_weight = utility_training_targets(table.iloc[order].reset_index(drop=True))
        np.testing.assert_array_equal(shuffled_target, target[order])
        np.testing.assert_allclose(shuffled_weight, weight[order])

    def test_normalized_weights_have_mean_one_and_same_ratios(self):
        table = cohort()
        _, raw = utility_training_targets(table, normalize=False)
        _, weight = utility_training_targets(table)
        self.assertAlmostEqual(weight.mean(), 1.0)
        np.testing.assert_allclose(weight * raw.mean(), raw)

    def test_input_table_is_not_modified(self):
        table = cohort()
        before = table.copy()
        utility_training_targets(table)
        pd.testing.assert_frame_equal(table, before)


def tabular(n, seed):
    rng = np.random.default_rng(seed)
    X = np.round(rng.normal(size=(n, 6)) * 3)  # few distinct values: binning is lossless
    y = (rng.random(n) < sigmoid(0.5 * X[:, 0] - 0.4 * X[:, 1] + (X[:, 2] > 1) - 2.5)).astype(int)
    weight = rng.exponential(size=n)
    return X, y, weight


class SampleWeightAgainstSklearn(unittest.TestCase):
    def test_logistic_regression(self):
        X, y, weight = tabular(4000, 0)
        ours = NumpyLogisticRegression(C=0.3).fit(X, y, sample_weight=weight)
        # Our "balanced" class weights come from row counts, as in scikit-learn's
        # trees. scikit-learn's logistic regression derives them differently
        # when sample weights are given, so the reference gets the product.
        balanced = np.where(y == 1, len(y) / (2.0 * y.sum()), len(y) / (2.0 * (len(y) - y.sum())))
        reference = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.3, solver="newton-cholesky", tol=1e-10, max_iter=1000),
        ).fit(X, y, logisticregression__sample_weight=weight * balanced)
        np.testing.assert_allclose(ours.coef_, reference[-1].coef_.ravel(), atol=1e-6)
        np.testing.assert_allclose(ours.intercept_, reference[-1].intercept_[0], atol=1e-6)

    def test_boosting_first_tree(self):
        X, y, weight = tabular(4000, 1)
        ours = NumpyGradientBoosting(max_iter=1).fit(X, y, sample_weight=weight)
        reference = HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=1, max_leaf_nodes=15, l2_regularization=2,
            class_weight="balanced", early_stopping=False,
        ).fit(X, y, sample_weight=weight)
        np.testing.assert_allclose(ours.decision_function(X), reference.decision_function(X), atol=1e-6)

    def test_single_forest_tree(self):
        X, y, weight = tabular(4000, 2)
        ours = NumpyRandomForest(n_estimators=1, bootstrap=False, max_features=None, max_depth=4).fit(
            X, y, sample_weight=weight)
        reference = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=0,
        ).fit(X, y, sample_weight=weight)
        np.testing.assert_allclose(ours.predict_proba(X), reference.predict_proba(X), atol=1e-10)


class SampleWeightBehaviour(unittest.TestCase):
    MODELS = {
        "logistic": lambda: NumpyLogisticRegression(class_weight=None),
        "boosting": lambda: NumpyGradientBoosting(max_iter=10, class_weight=None),
        "forest": lambda: NumpyRandomForest(n_estimators=5, class_weight=None),
    }

    def test_unit_weights_change_nothing(self):
        X, y, _ = tabular(2000, 3)
        for name, make in self.MODELS.items():
            with self.subTest(name):
                np.testing.assert_allclose(
                    make().fit(X, y, sample_weight=np.ones(len(y))).decision_function(X),
                    make().fit(X, y).decision_function(X), atol=1e-12,
                )

    def test_zero_weight_rows_do_not_influence_the_fit(self):
        X, y, weight = tabular(2000, 4)
        ignored = np.arange(len(y)) % 5 == 0
        weight[ignored] = 0.0
        flipped = np.where(ignored, 1 - y, y)  # only the ignored rows' labels change
        for name, make in self.MODELS.items():
            with self.subTest(name):
                np.testing.assert_allclose(
                    make().fit(X, flipped, sample_weight=weight).decision_function(X),
                    make().fit(X, y, sample_weight=weight).decision_function(X), atol=1e-8,
                )

    def test_upweighting_positives_raises_predicted_risk(self):
        X, y, _ = tabular(2000, 5)
        for name, make in self.MODELS.items():
            with self.subTest(name):
                plain = make().fit(X, y).predict_proba(X)[:, 1].mean()
                heavy = make().fit(X, y, sample_weight=np.where(y == 1, 10.0, 1.0)).predict_proba(X)[:, 1].mean()
                self.assertGreater(heavy, plain + 0.05)

    def test_bad_weights_are_rejected(self):
        X, y, weight = tabular(500, 6)
        for name, make in self.MODELS.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    make().fit(X, y, sample_weight=weight[:-1])
                with self.assertRaises(ValueError):
                    make().fit(X, y, sample_weight=-weight)
                with self.assertRaises(ValueError):
                    make().fit(X, y, sample_weight=np.where(y == 1, 0.0, 1.0))

    def test_models_fit_on_utility_targets(self):
        rng = np.random.default_rng(7)
        stays = [stay(f"s{i}", 40, label_from=int(rng.integers(8, 30))) for i in range(30)]
        stays += [stay(f"n{i}", 40) for i in range(120)]
        table = pd.concat(stays, ignore_index=True)
        target, weight = utility_training_targets(table)
        # One noisy feature that rises as the utility-preferred alarm window opens.
        X = (target + rng.normal(scale=0.7, size=len(table)))[:, None]
        self.assertGreater(target.sum(), table["label"].sum() * 0.5)
        for name, make in self.MODELS.items():
            with self.subTest(name):
                score = make().fit(X, target, sample_weight=weight).decision_function(X)
                self.assertTrue(np.all(np.isfinite(score)))
                self.assertGreater(score[target == 1].mean(), score[target == 0].mean())


if __name__ == "__main__":
    unittest.main()
