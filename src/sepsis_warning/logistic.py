"""Logistic regression and Platt calibration in pure NumPy (K01, K04).

No scikit-learn is imported here. The scikit-learn models in ``modeling.py``
stay as the reference benchmark; ``tests/test_logistic.py`` checks this
module against them.

The logistic regression minimises the same objective as scikit-learn,

    0.5 * ||w||^2 + C * sum_i s_i * logloss(y_i, x_i @ w + b)

with the intercept ``b`` unpenalised, using full-batch Newton steps (IRLS,
Bishop PRML 4.3.3). The problem is convex, so the fit is deterministic and
needs no learning rate, seed or epoch count.

The shared helpers here (``sigmoid``, ``row_weights``, input checks and
``calibrate_frozen``) are also used by ``random_forest.py`` and
``gradient_boosting.py``.
"""

import warnings

import numpy as np


def _as_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
    return X


def sigmoid(z):
    """Logistic function that never exponentiates a positive number."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _log_loss_terms(z, target):
    """Per-row cross-entropy from logits; ``target`` may be a soft label."""
    return np.logaddexp(0.0, z) - target * z


def _binary_labels(y):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    y = y.astype(np.float64)
    if y.min() == y.max():
        raise ValueError("labels contain a single class")
    return y


def row_weights(y, class_weight, sample_weight=None):
    """Per-row loss weights: class weights times optional ``sample_weight``.

    ``"balanced"`` class weights are ``n / (2 * n_class)`` from row counts.
    """
    n = len(y)
    if class_weight is None:
        weight = np.ones(n)
    else:
        positives = y.sum()
        weight = np.where(y == 1, n / (2.0 * positives), n / (2.0 * (n - positives)))
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape != (n,):
            raise ValueError("sample_weight must have one value per row")
        if not np.isfinite(sample_weight).all() or (sample_weight < 0).any():
            raise ValueError("sample_weight must be finite and non-negative")
        weight = weight * sample_weight
        if not (weight[y == 1].sum() > 0 and weight[y == 0].sum() > 0):
            raise ValueError("both classes need some positive sample_weight")
    return weight


class MedianImputer:
    """Fill missing values with training medians. All-missing columns get 0."""

    def fit(self, X):
        X = _as_matrix(X)
        self.medians_ = np.zeros(X.shape[1])
        seen = ~np.isnan(X).all(axis=0)
        if seen.any():
            self.medians_[seen] = np.nanmedian(X[:, seen], axis=0)
        return self

    def transform(self, X):
        X = _as_matrix(X)
        return np.where(np.isnan(X), self.medians_, X)


class StandardScaler:
    """Centre and scale with training mean and standard deviation.

    Constant columns keep scale 1 so they become exactly zero.
    """

    def fit(self, X):
        X = _as_matrix(X)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, X):
        return (_as_matrix(X) - self.mean_) / self.scale_


class NumpyLogisticRegression:
    """Median-impute, standardise, then L2 logistic regression by Newton steps.

    Preprocessing statistics come from the rows passed to ``fit`` only.

    Parameters
    ----------
    C : inverse regularisation strength, same meaning as scikit-learn.
    class_weight : ``"balanced"`` (weights ``n / (2 * n_class)``) or ``None``.
    max_iter, tol : Newton stops when the largest gradient entry of the
        mean weighted loss falls below ``tol``.
    chunk_rows : rows per block when accumulating the Hessian, to bound memory.
    """

    def __init__(self, C=1.0, class_weight="balanced", max_iter=100, tol=1e-8, chunk_rows=65536):
        if C <= 0:
            raise ValueError("C must be positive")
        if class_weight not in ("balanced", None):
            raise ValueError("class_weight must be 'balanced' or None")
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.chunk_rows = chunk_rows

    def _objective(self, Z, y, s, theta):
        z = Z @ theta[:-1] + theta[-1]
        return 0.5 * theta[:-1] @ theta[:-1] + self.C * (s @ _log_loss_terms(z, y)), z

    def _hessian(self, Z, d):
        """X1' diag(d) X1 for X1 = [Z, 1], accumulated in row blocks."""
        n, k = Z.shape
        H = np.zeros((k + 1, k + 1))
        for start in range(0, n, self.chunk_rows):
            block, weight = Z[start:start + self.chunk_rows], d[start:start + self.chunk_rows]
            weighted = block * weight[:, None]
            H[:k, :k] += block.T @ weighted
            H[:k, k] += weighted.sum(axis=0)
            H[k, k] += weight.sum()
        H[k, :k] = H[:k, k]
        return H

    def fit(self, X, y, sample_weight=None):
        """Fit on rows ``X`` with labels ``y``.

        ``sample_weight`` is an optional non-negative weight per row. It
        multiplies the class weights, as in scikit-learn, and does not affect
        the imputation medians or scaling statistics.
        """
        X = _as_matrix(X)
        y = _binary_labels(y)
        if len(y) != len(X):
            raise ValueError("X and y have different numbers of rows")
        self.imputer_ = MedianImputer().fit(X)
        Z = self.imputer_.transform(X)
        self.scaler_ = StandardScaler().fit(Z)
        Z = self.scaler_.transform(Z)

        s = row_weights(y, self.class_weight, sample_weight)
        k = Z.shape[1]
        penalty = np.append(np.ones(k), 0.0)  # the intercept is not penalised
        theta = np.zeros(k + 1)
        loss, z = self._objective(Z, y, s, theta)
        grad_scale = self.C * s.sum()
        self.converged_ = False
        self.loss_history_ = [loss]

        for iteration in range(1, self.max_iter + 1):
            p = sigmoid(z)
            residual = s * (p - y)
            grad = penalty * theta + self.C * np.append(Z.T @ residual, residual.sum())
            if np.abs(grad).max() / grad_scale <= self.tol:
                self.converged_ = True
                iteration -= 1
                break
            H = self.C * self._hessian(Z, s * p * (1.0 - p)) + np.diag(penalty)
            # A tiny ridge on the intercept keeps the solve well-posed if every
            # p*(1-p) underflows; it does not change a well-conditioned step.
            H[k, k] += 1e-12 * grad_scale
            step = np.linalg.solve(H, grad)

            # Backtracking keeps each step a descent step (Armijo condition).
            slope, scale = grad @ step, 1.0
            while True:
                candidate = theta - scale * step
                new_loss, new_z = self._objective(Z, y, s, candidate)
                if new_loss <= loss - 1e-4 * scale * slope or scale < 1e-10:
                    break
                scale *= 0.5
            theta, loss, z = candidate, new_loss, new_z
            self.loss_history_.append(loss)

        self.n_iter_ = iteration
        if not self.converged_:
            p = sigmoid(z)
            residual = s * (p - y)
            grad = penalty * theta + self.C * np.append(Z.T @ residual, residual.sum())
            self.converged_ = bool(np.abs(grad).max() / grad_scale <= self.tol)
        if not self.converged_:
            warnings.warn(
                f"NumpyLogisticRegression did not converge in {self.max_iter} Newton iterations",
                RuntimeWarning, stacklevel=2,
            )
        self.coef_ = theta[:-1].copy()
        self.intercept_ = float(theta[-1])
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        Z = self.scaler_.transform(self.imputer_.transform(X))
        return Z @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])


class PlattCalibrator:
    """Platt (1999) scaling: ``P(y=1 | z) = sigmoid(a * z + b)``.

    ``a`` and ``b`` are fitted by Newton steps on the calibration logits, with
    Platt's smoothed targets ``(n_pos + 1) / (n_pos + 2)`` and ``1 / (n_neg + 2)``.
    Rows are unweighted, so the output follows the natural hourly prevalence.
    """

    def __init__(self, max_iter=100, tol=1e-10):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, logits, y):
        z = np.asarray(logits, dtype=np.float64).ravel()
        y = _binary_labels(y)
        if len(z) != len(y):
            raise ValueError("logits and y have different lengths")
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        target = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        design = np.column_stack([z, np.ones_like(z)])

        def mean_loss(params):
            return _log_loss_terms(design @ params, target).mean()

        params = np.array([0.0, np.log((n_pos + 1.0) / (n_neg + 1.0))])
        loss = mean_loss(params)
        self.converged_ = False
        for iteration in range(1, self.max_iter + 1):
            p = sigmoid(design @ params)
            grad = design.T @ (p - target) / len(z)
            if np.abs(grad).max() <= self.tol:
                self.converged_ = True
                iteration -= 1
                break
            H = (design * (p * (1.0 - p))[:, None]).T @ design / len(z)
            H += 1e-12 * np.eye(2)
            step = np.linalg.solve(H, grad)
            slope, scale = grad @ step, 1.0
            while True:
                candidate = params - scale * step
                new_loss = mean_loss(candidate)
                if new_loss <= loss - 1e-4 * scale * slope or scale < 1e-10:
                    break
                scale *= 0.5
            params, loss = candidate, new_loss
        self.n_iter_ = iteration
        if not self.converged_:
            p = sigmoid(design @ params)
            self.converged_ = bool(np.abs(design.T @ (p - target) / len(z)).max() <= self.tol)
        if not self.converged_:
            warnings.warn(
                f"PlattCalibrator did not converge in {self.max_iter} Newton iterations",
                RuntimeWarning, stacklevel=2,
            )
        self.a_, self.b_ = float(params[0]), float(params[1])
        return self

    def predict(self, logits):
        return sigmoid(self.a_ * np.asarray(logits, dtype=np.float64) + self.b_)


class CalibratedNumpyModel:
    """A frozen base model plus its Platt calibrator, with ``predict_proba``."""

    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = self.calibrator.predict(self.base.decision_function(X))
        return np.column_stack([1.0 - p, p])


def calibrate_frozen(base, calibration, columns):
    """Platt-calibrate an already fitted model without refitting it.

    ``base`` needs ``decision_function``; the calibrator sees only its logits
    on the ``calibration`` rows.
    """
    calibrator = PlattCalibrator().fit(base.decision_function(calibration[columns]), calibration["label"])
    return CalibratedNumpyModel(base, calibrator)


def fit_numpy_calibrated(train, calibration, columns, C=0.3):
    """NumPy counterpart of ``modeling.fit_calibrated`` for logistic regression.

    The base model is fitted on ``train`` only, then calibrated on ``calibration``.
    """
    base = NumpyLogisticRegression(C=C).fit(train[columns], train["label"])
    return calibrate_frozen(base, calibration, columns)
