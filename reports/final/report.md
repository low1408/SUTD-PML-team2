# Early sepsis warning on the PhysioNet 2019 challenge data

**Course 61.501 Production Ready ML. Team 2.** Draft skeleton: every `TODO` is for a named owner to fill from the
evidence listed under it. Do not paste numbers that were not produced by the commands in the "Reproduce" box.

> Results folder: `reports/results/` (made by `model-report`; see `FIGURES.md` there for each figure's recreate
> command). Never paste patient-level rows, real-data ward pages or the project brief (it contains a cluster password).

## 1. Executive Summary (at most one page)

We warn ICU teams early that a patient is likely to develop sepsis, using hourly vital signs and labs from the
PhysioNet/CinC 2019 challenge. A random forest (276 causal features per hour, Platt-calibrated, threshold 0.05, 6 h
cooldown) was chosen on a validation split by the one-standard-error rule, trained and evaluated on AWS, and scored once
on a frozen internal test split and on a second hospital.

At the budget of 0.2 false alerts per non-septic patient-day, validation utility is 0.379 (95% interval 0.331 to 0.431)
against 0.018 for the SIRS rule held to the same budget; early-detection recall is 56.1% against 10.2%, with a median
warning lead of 9 h. On the internal test split utility is 0.351, and on the unseen hospital B (20,000 stays) it is 0.255
against -0.014 for SIRS, with 0.140 false alerts per non-septic patient-day, so the budget held. Performance drops across
hospitals (early recall 47.0% at B). Main limitation: retrospective data, a benchmark label that is not clinical truth,
and low precision at this prevalence (average precision 0.062 at B).

TODO (leo): add precision, recall and F1 once the per-stay predictions are exported (see 7.1); re-read this last.

## 2. Background / Introduction

TODO (Mashie): why early sepsis warning matters, what the 2019 challenge asks, why alert budget and time-to-alert
matter more than accuracy. Evidence: `docs/data-profile.md`, `docs/clinical-use-design-and-validation-plan.md`.

## 3. Related work

TODO (Kaiyuan): the PhysioNet/CinC 2019 challenge and its utility score; SIRS and qSOFA rules; published sepsis
early-warning models (cite each; say what we reuse and what we change). Keep to sources the team has read.

## 4. Problem formulation and overview

- Task: hourly risk score per ICU stay; label already shifted 6 hours before onset; alert when the calibrated
  probability crosses a threshold chosen once on validation, then a 6-hour cooldown.
- Constraint: at most 0.2 false alert events per non-septic patient-day on validation.
- Splits: hospital A 55/15/15/15 (train, calibration, validation, internal test); hospital B is the external test.
  Test splits are scored once, after the model is frozen.
- Pipeline diagram: TODO (Josh) export from `docs/presentation/` or the repository's pipeline page into
  `reports/final/pipeline.png`.

## 5. Data description

TODO (Jawad): sources, counts of stays and hours per hospital and split, prevalence, missingness, units, the
cleaning rules and what was excluded. Evidence: `docs/data-profile.md`, `reports/figures/` (cohort prevalence, stay
length, age and gender, missingness, vital signs), `docs/hospital-a-vs-b.md`.

## 6. Details of the solution

- Code: https://github.com/low1408/SUTD-PML-team2 (commit: TODO fill from `git rev-parse HEAD`).
- Features: 276 causal features per hour (`src/sepsis_warning/features.py`); no future information.
- Models: constant prior, logistic regression, random forest and histogram gradient boosting (the real-data run on AWS
  did not include the hand-written NumPy models). Selection rule: highest validation utility, then the simplest model
  within one standard error. Result: histogram gradient boosting scored 0.383 (standard error 0.0257) and random forest
  0.379 (0.0259), so the simpler random forest was selected (threshold 0.05, cooldown 6 h).
- Hyperparameters: reference settings in `src/sepsis_warning/models.toml`; search space in
  `tuning_space.toml`; search method: random/grid sample, patient-grouped (stratified) cross-validation on the
  training split only, selected by mean cross-validated average precision, logged to MLflow experiment
  `sepsis-warning-tuning`. The AWS run used 3 patient-grouped folds, 6 trials per model, 11,183 training stays and seed
  42. Best cross-validated average precision: histogram gradient boosting 0.1069, random forest 0.0903, logistic
  regression 0.0855. TODO (Edwin): paste the chosen values from `tuned_models.toml` and the figure `fig_tuning.png`.
- Calibration and threshold: Platt scaling on a separate calibration split (3,051 stays) takes the Brier score from
  0.1371 (class-weighted, uncalibrated) to 0.0201, below the constant-prior 0.0213. The threshold is the lowest one on
  the validation split (3,051 stays) that keeps false alerts at or below 0.2 per non-septic patient-day. TODO (Kaiyuan): check wording.
- Engineering (MLflow registry, gates, API, drift monitoring, AWS): one paragraph plus `docs/mlops-audit.md`.

### Reproduce

```bash
python -m sepsis_warning split --data <combined.parquet>
python -m sepsis_warning tune  --data <combined.parquet> --output artifacts/tuning --mlflow-uri sqlite:///mlflow.db
python -m sepsis_warning train --data <combined.parquet> --output artifacts/real --models-config artifacts/tuning/tuned_models.toml --include-numpy-models
python -m sepsis_warning model-report --run artifacts/real --output reports/results --tuning artifacts/tuning
# only after the team freezes the model:
python -m sepsis_warning evaluate --data <combined.parquet> --model artifacts/real/model.joblib --output artifacts/evaluation
python -m sepsis_warning model-report --run artifacts/real --output reports/results_test --evaluation artifacts/evaluation
```

Seeds, package versions and hardware: `environment.json` next to the results.

## 7. Evaluation

### 7.1 Results against the baseline

Random forest (frozen, threshold 0.05, 6 h cooldown), from the MLflow runs on AWS. Budget: at most 0.2 false alerts per
non-septic patient-day.

| Metric | Validation (3,051 stays) | Internal test (3,051) | Hospital B test (20,000) |
|---|---:|---:|---:|
| Normalized hourly utility | 0.379 (CI 0.331 to 0.431) | 0.351 | 0.255 |
| False alerts per non-septic patient-day | 0.177 | 0.174 | 0.140 |
| Early-detection recall | 56.1% | 54.3% | 47.0% |
| Median warning lead time | 9 h | 9 h | 9 h |
| AUROC | 0.817 | 0.807 | 0.792 |
| Average precision | 0.121 (prior 0.022) | 0.099 (prior 0.022) | 0.062 (prior 0.014) |
| Brier score | 0.0201 | 0.0202 | 0.0136 |

SIRS rule: on validation, held to the same budget, utility 0.018 and early recall 10.2%. At its standard two-criteria
setting it raises 1.972 false alerts per non-septic patient-day on the internal test and 1.518 at hospital B (8 to 10
times the budget), with utility 0.087 and -0.014 and early recall 75.9% and 75.5%.

Intervals are reported for validation only; TODO (leo) add test intervals if available. Precision, recall and F1 are
not in the AWS output (only aggregate metrics are exported), so they must come from a local run or be left out.
Accuracy is near-trivial: a model that never alerts scores 97.8% at hospital A and 98.6% at B with utility 0.
Report accuracy, precision, recall, F1, average precision, hourly utility and the intervals. State plainly that accuracy
is near-trivial at this prevalence.

### 7.2 Plots

TODO (leo): precision-recall curves (`fig_pr_curves.png`), model comparison (`fig_metrics_bars.png`),
threshold trade-off, calibration, confusion matrices, loss curves where a model has them. Each caption quotes the
figure's recreate command from `FIGURES.md`.

### 7.3 Important features

TODO (Jawad): permutation or model-based importance for the selected model, with the clinical sense-check.
Evidence: `feature_importance.md` and `fig_feature_importance.png` in the results folder (grouped permutation
importance on validation). Say how much comes from `measurement_pattern` (whether and when something was measured)
and from `ICULOS`, because those reflect care practice and time in the ICU rather than physiology.

### 7.4 Bias audit

TODO (Mashie): performance by hospital, age band and gender; report gaps with intervals; say what was not checked.
Evidence: `subgroup_table.md` (validation) and, after the freeze, `test_subgroup_table.md` (sex and age band within
each test split, then hospital A against hospital B), `docs/hospital-a-vs-b.md`,
`src/sepsis_warning/hospital_comparison.py`.

### 7.5 Where the model goes wrong

TODO (Josh): two or three anonymised failure cases (`fig_failure_examples.png`; keep private if from real data)
and the SIRS failure patterns in `docs/sirs-failure-patterns.md`.

## 8. Discussion

TODO (all): what worked, what surprised us, why the model beats or does not beat SIRS, how much of the gap is the
alert budget, hospital A to B shift, what the intervals say about the team's claims.

## 9. Recommendations

TODO (leo): what a hospital should do before use (silent trial, threshold re-tuning on local data, monitoring,
human review), tied to `docs/clinical-use-design-and-validation-plan.md` and `docs/monitoring-plan.md`.

## 10. Limitations and future work

TODO (all): benchmark label is not clinical truth; retrospective data only; two hospitals; no real-time latency
data; calibration drift; fairness checks are partial. Future: more sites, prospective trial, richer features.

## 11. Optional: future research

TODO (optional): sequence models were out of scope (no deep learning); other directions the team finds worthwhile.

---

## Member contributions

| Member | Contribution (TODO confirm with each person) |
|---|---|
| Edwin | |
| Jawad | |
| Kaiyuan | |
| Josh | |
| leo | |
| Mashie | |

## Rubric checklist (delete before submitting)

- [ ] Concept: problem and approach clear in section 1 and 4
- [ ] Coding: code link, tests passing, reproduce commands run end to end
- [ ] Performance and evaluation: hyperparameter search shown, baseline comparison, accuracy, precision, recall, F1, PR curves
- [ ] Creativity: alert-budget utility, one-standard-error selection, freeze-before-test, production stack
- [ ] Report 30%: all 11 sections, figures captioned with recreate commands
- [ ] Brief and passwords not committed
