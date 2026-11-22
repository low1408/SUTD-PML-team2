"""Candidate models from a versioned config, with calibration on separate patients.

The candidates, their settings and their complexity rank live in
``models.toml`` (package data). Its SHA-256 goes into every run's metadata and
the model manifest, so a settings change is visible in the lineage.
"""

import hashlib
import tomllib
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG = Path(__file__).with_name("models.toml")


def _numpy_logistic(seed, params):
    from .logistic import NumpyLogisticRegression

    return NumpyLogisticRegression(**params)  # deterministic: Newton steps, no random state


def _numpy_forest(seed, params):
    from .random_forest import NumpyRandomForest

    return NumpyRandomForest(seed=seed, **params)


def _numpy_boosting(seed, params):
    from .gradient_boosting import NumpyGradientBoosting

    return NumpyGradientBoosting(seed=seed, **params)


def _imputed(*steps):
    # Medians are learned inside the pipeline, so from training rows only.
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), *steps)


BUILDERS = {
    "sklearn.constant_prior": lambda seed, params: DummyClassifier(strategy="prior"),
    "sklearn.logistic_regression": lambda seed, params: _imputed(
        StandardScaler(), LogisticRegression(random_state=seed, **params)),
    "sklearn.random_forest": lambda seed, params: _imputed(RandomForestClassifier(random_state=seed, **params)),
    "sklearn.hist_gradient_boosting": lambda seed, params: HistGradientBoostingClassifier(random_state=seed, **params),
    "numpy.logistic_regression": _numpy_logistic,
    "numpy.random_forest": _numpy_forest,
    "numpy.gradient_boosting": _numpy_boosting,
}


def load_model_config(path: Path | None = None) -> tuple[dict, str]:
    """Return (config, SHA-256 of the file). Rejects unknown kinds and missing ranks."""
    path = Path(path or DEFAULT_CONFIG)
    raw = path.read_bytes()
    config = tomllib.loads(raw.decode("utf-8"))
    if not config:
        raise ValueError(f"{path} defines no models")
    for name, entry in config.items():
        if entry.get("kind") not in BUILDERS:
            raise ValueError(f"{path}: model {name!r} has unknown kind {entry.get('kind')!r}")
        if not isinstance(entry.get("complexity"), int):
            # A config error like the others above, so ValueError rather than TypeError.
            raise ValueError(f"{path}: model {name!r} needs an integer complexity")  # noqa: TRY004
    return config, hashlib.sha256(raw).hexdigest()


def candidate_models(seed, config: dict | None = None, include_numpy: bool = False) -> dict:
    """Unfitted estimators by name: enabled entries, plus NumPy ones when asked."""
    if config is None:
        config, _ = load_model_config()
    models = {}
    for name, entry in config.items():
        wanted = entry.get("enabled", True) or (include_numpy and entry["kind"].startswith("numpy."))
        if wanted:
            models[name] = BUILDERS[entry["kind"]](seed, dict(entry.get("params", {})))
    if not models:
        raise ValueError("no candidate model is enabled")
    return models


def complexity_ranks(config: dict | None = None) -> dict:
    if config is None:
        config, _ = load_model_config()
    return {name: entry["complexity"] for name, entry in config.items()}


def is_numpy_model(estimator) -> bool:
    return type(estimator).__module__.startswith("sepsis_warning.")


def fit_calibrated(estimator, train, calibration, columns, is_constant=False):
    estimator.fit(train[columns], train["label"])
    if is_constant:
        return estimator
    if is_numpy_model(estimator):
        from .logistic import calibrate_frozen

        # Same idea as below: the fitted model is frozen, only Platt scaling sees calibration rows.
        return calibrate_frozen(estimator, calibration, columns)
    # FrozenEstimator prevents retraining the underlying model on calibration
    # patients. Calibration itself is fitted at natural hourly prevalence.
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    calibrated.fit(calibration[columns], calibration["label"])
    return calibrated
