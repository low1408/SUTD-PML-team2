"""Small classical model comparison with calibration on separate patients."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def candidate_models(seed):
    return {
        "constant_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=80, max_depth=8, min_samples_leaf=20,
                class_weight="balanced", random_state=seed, n_jobs=2,
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=100, max_leaf_nodes=15,
            l2_regularization=2, class_weight="balanced", random_state=seed,
            # Avoid an automatic row-level validation split within patients.
            early_stopping=False,
        ),
    }


def fit_calibrated(estimator, train, calibration, columns, is_constant=False):
    estimator.fit(train[columns], train["label"])
    if is_constant:
        return estimator
    # FrozenEstimator prevents retraining the underlying model on calibration
    # patients. Calibration itself is fitted at natural hourly prevalence.
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    calibrated.fit(calibration[columns], calibration["label"])
    return calibrated
