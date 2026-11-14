# NumPy models vs scikit-learn reference (validation only)

A local check run on 14 Nov 2026, after PR #68 and #69 were merged. It is not
the official K01–K03 result: the models are Kaiyuan's, and the official numbers
come from `train` once the NumPy models are wired in.

- **Data:** PhysioNet `combined.parquet` with the saved `splits.csv` (seed 42).
- **Fit:** train split (435,449 hours), Platt calibration on the calibration split
  (117,195 hours), 276 features from the shared builder, 6-hour window.
- **Scored on:** hospital A validation only (3,051 stays, 268 septic, 118,668 hours).
  Internal test and hospital B were not built or scored.
- **Policy:** same `tune_threshold` as `train`: at most 0.20 false alarms per
  non-septic patient-day, 6-hour cooldown.
- **Rerun:** `python scripts/check_numpy_models.py` (about 20 minutes; add
  `--numpy-only` to skip the scikit-learn references). Seconds depend on the machine
  (here an Intel i9-14900HX, 2 threads).

| Model | Utility | AUROC | AP | False alarms / patient-day | Septic stays warned early | Threshold |
|---|---|---|---|---|---|---|
| SIRS rule (from `reports/baselines/`) | 0.084 | 0.615 | 0.032 | 1.98 | 77.9% | ≥ 2 criteria |
| NumPy logistic regression | 0.250 | 0.795 | 0.105 | 0.151 | 47.1% | 0.06 |
| scikit-learn logistic regression | 0.250 | 0.795 | 0.105 | 0.151 | 47.1% | 0.06 |
| NumPy random forest | 0.334 | 0.818 | 0.116 | 0.140 | 51.6% | 0.07 |
| scikit-learn random forest | 0.365 | 0.817 | 0.125 | 0.176 | 57.8% | 0.06 |
| NumPy gradient boosting | **0.398** | 0.831 | **0.139** | 0.185 | 57.8% | 0.06 |
| scikit-learn gradient boosting | 0.390 | **0.833** | 0.136 | 0.192 | 59.8% | 0.06 |

Every model stays under the 0.20 alarm budget and scores 3–4.7 times SIRS on utility.
The NumPy logistic regression matches scikit-learn exactly, and the NumPy gradient
boosting is slightly ahead of it. The NumPy random forest picks a stricter threshold, so
it raises fewer alarms but warns fewer septic stays.

`validation_numpy_vs_sklearn.csv` holds every metric. All values are aggregates; there are
no patient-level rows.
