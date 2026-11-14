"""Pure-NumPy logistic regression and Platt scaling against scikit-learn.

Synthetic data only: these tests check the optimiser and the calibration
maths, not real-cohort performance.

Run: MPLBACKEND=Agg .venv/bin/python -m unittest tests.test_logistic -v
"""

import ast
import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler as SklearnScaler

from sepsis_warning import logistic
from sepsis_warning.logistic import (
    MedianImputer, NumpyLogisticRegression, PlattCalibrator, StandardScaler,
    fit_numpy_calibrated, sigmoid,
)


def synthetic(n, seed, features=12):
    """Imbalanced, correlated features with missing values, one empty column
    and one constant column."""
    # The mixing matrix and true weights are shared by every table, so tables
    # built with different seeds are draws from one population.
    shared = np.random.default_rng(0)
    mixing = shared.normal(size=(features, features)) / np.sqrt(features)
    truth = shared.normal(size=features)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, features)) @ mixing
    y = (rng.random(n) < sigmoid(X @ truth - 4.0)).astype(int)
    X[rng.random(X.shape) < 0.2] = np.nan
    X = np.column_stack([X, np.full(n, np.nan), np.full(n, 3.0)])
    columns = [f"f{i}" for i in range(X.shape[1])]
    table = pd.DataFrame(X, columns=columns)
    table["label"] = y
    return table, columns


def sklearn_reference(C):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        SklearnScaler(),
        LogisticRegression(C=C, class_weight="balanced", solver="newton-cholesky", tol=1e-10, max_iter=1000),
    )


class Preprocessing(unittest.TestCase):
    def test_imputer_and_scaler_match_sklearn(self):
        table, columns = synthetic(500, 1)
        X = table[columns].to_numpy()
        imputed = MedianImputer().fit(X).transform(X)
        expected = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
        np.testing.assert_allclose(imputed, expected)
        np.testing.assert_allclose(
            StandardScaler().fit(imputed).transform(imputed), SklearnScaler().fit_transform(expected), atol=1e-12,
        )

    def test_statistics_come_from_fit_rows_only(self):
        train = np.array([[1.0], [3.0], [np.nan]])
        imputer = MedianImputer().fit(train)
        np.testing.assert_array_equal(imputer.transform(np.array([[np.nan], [100.0]])), [[2.0], [100.0]])

    def test_sigmoid_is_silent_and_exact_at_extremes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            values = sigmoid(np.array([-1e4, -800.0, 0.0, 800.0, 1e4]))
        np.testing.assert_array_equal(values, [0.0, 0.0, 0.5, 1.0, 1.0])


class LogisticRegressionAgainstSklearn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, cls.columns = synthetic(6000, 2)
        cls.model = NumpyLogisticRegression(C=0.3).fit(cls.table[cls.columns], cls.table["label"])
        cls.reference = sklearn_reference(0.3).fit(cls.table[cls.columns], cls.table["label"])

    def test_converges_in_few_newton_steps(self):
        self.assertTrue(self.model.converged_)
        self.assertLess(self.model.n_iter_, 30)
        self.assertTrue(np.all(np.diff(self.model.loss_history_) <= 0))

    def test_coefficients_match(self):
        fitted = self.reference[-1]
        np.testing.assert_allclose(self.model.coef_, fitted.coef_.ravel(), atol=1e-6)
        np.testing.assert_allclose(self.model.intercept_, fitted.intercept_[0], atol=1e-6)

    def test_probabilities_match_on_new_rows(self):
        fresh, _ = synthetic(1000, 3)
        np.testing.assert_allclose(
            self.model.predict_proba(fresh[self.columns]),
            self.reference.predict_proba(fresh[self.columns]), atol=1e-6,
        )

    def test_unweighted_fit_matches(self):
        model = NumpyLogisticRegression(C=2.0, class_weight=None).fit(self.table[self.columns], self.table["label"])
        reference = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), SklearnScaler(),
            LogisticRegression(C=2.0, solver="newton-cholesky", tol=1e-10, max_iter=1000),
        ).fit(self.table[self.columns], self.table["label"])
        np.testing.assert_allclose(model.coef_, reference[-1].coef_.ravel(), atol=1e-6)

    def test_row_chunking_does_not_change_the_fit(self):
        chunked = NumpyLogisticRegression(C=0.3, chunk_rows=97).fit(self.table[self.columns], self.table["label"])
        np.testing.assert_allclose(chunked.coef_, self.model.coef_, atol=1e-10)

    def test_empty_and_constant_columns_get_zero_weight(self):
        np.testing.assert_array_equal(self.model.coef_[-2:], [0.0, 0.0])

    def test_non_convergence_is_reported(self):
        with self.assertWarns(RuntimeWarning):
            model = NumpyLogisticRegression(C=0.3, max_iter=1).fit(self.table[self.columns], self.table["label"])
        self.assertFalse(model.converged_)

    def test_rejects_bad_labels(self):
        X = self.table[self.columns]
        with self.assertRaises(ValueError):
            NumpyLogisticRegression().fit(X, np.zeros(len(X)))
        with self.assertRaises(ValueError):
            NumpyLogisticRegression().fit(X, np.full(len(X), 2))


class PlattCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.columns = synthetic(6000, 4)
        cls.calibration, _ = synthetic(3000, 5)
        cls.validation, _ = synthetic(3000, 6)
        cls.model = fit_numpy_calibrated(cls.train, cls.calibration, cls.columns, C=0.3)

    def test_matches_sklearn_sigmoid_calibration(self):
        base = sklearn_reference(0.3).fit(self.train[self.columns], self.train["label"])
        reference = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        reference.fit(self.calibration[self.columns], self.calibration["label"])
        np.testing.assert_allclose(
            self.model.predict_proba(self.validation[self.columns]),
            reference.predict_proba(self.validation[self.columns]), atol=1e-4,
        )

    def test_calibration_restores_natural_prevalence(self):
        columns, label = self.columns, self.validation["label"]
        raw = self.model.base.predict_proba(self.validation[columns])[:, 1]
        calibrated = self.model.predict_proba(self.validation[columns])[:, 1]
        self.assertGreater(raw.mean(), 3 * label.mean())  # balanced weights inflate risk
        self.assertAlmostEqual(calibrated.mean(), label.mean(), delta=0.02)
        self.assertLess(np.mean((calibrated - label) ** 2), np.mean((raw - label) ** 2))

    def test_calibration_keeps_ranking_and_base_model(self):
        columns = self.columns
        before = self.model.base.coef_.copy()
        logits = self.model.base.decision_function(self.validation[columns])
        calibrated = self.model.predict_proba(self.validation[columns])[:, 1]
        self.assertGreater(self.model.calibrator.a_, 0)
        np.testing.assert_array_equal(np.argsort(logits, kind="stable"), np.argsort(calibrated, kind="stable"))
        np.testing.assert_array_equal(self.model.base.coef_, before)

    def test_calibrator_converges(self):
        self.assertTrue(self.model.calibrator.converged_)

    def test_single_class_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            PlattCalibrator().fit(np.array([0.1, 0.2]), np.array([0, 0]))

    def test_joblib_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(self.model, path)
            loaded = joblib.load(path)
        np.testing.assert_array_equal(
            loaded.predict_proba(self.validation[self.columns]),
            self.model.predict_proba(self.validation[self.columns]),
        )


class NoSklearn(unittest.TestCase):
    def test_module_imports_only_numpy_and_stdlib(self):
        tree = ast.parse(Path(logistic.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"numpy", "warnings"})


if __name__ == "__main__":
    unittest.main()
