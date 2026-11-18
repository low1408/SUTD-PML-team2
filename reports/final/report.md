# Early sepsis warning on the PhysioNet 2019 challenge data

**Course 61.501 Production Ready ML. Team 2.** Draft skeleton: every `TODO` is for a named owner to fill from the
evidence listed under it. Do not paste numbers that were not produced by the commands in the "Reproduce" box.

> Results folder: `reports/results/` (made by `model-report`; see `FIGURES.md` there for each figure's recreate
> command). Never paste patient-level rows, real-data ward pages or the project brief (it contains a cluster password).

## 1. Executive Summary (at most one page)

TODO (leo): problem in one sentence, approach in two, the headline result against the SIRS baseline
(precision, recall, F1, hourly utility at 0.2 false alerts per non-septic patient-day) with its interval, the
external hospital B result, and the main limitation. Write this last.

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
- Models: constant prior, logistic regression, random forest, histogram gradient boosting, and the hand-written
  NumPy models (`--include-numpy-models`). Selection rule: highest validation utility, then the simplest model
  within one standard error.
- Hyperparameters: reference settings in `src/sepsis_warning/models.toml`; search space in
  `tuning_space.toml`; search method: random/grid sample, patient-grouped (stratified) cross-validation on the
  training split only, selected by mean cross-validated average precision, logged to MLflow experiment
  `sepsis-warning-tuning`. TODO (Edwin): paste the chosen values from `tuned_models.toml` and the cross-validation
  figure `fig_tuning.png`.
- Calibration and threshold: TODO (Kaiyuan) calibration method and how the threshold is tuned.
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

TODO (leo): paste `reports/results/metrics_table.md` (validation) and, after the freeze, `test_metrics_table.md`.
Report accuracy, precision, recall, F1, average precision, hourly utility and the intervals. State plainly that accuracy
is near-trivial at this prevalence.

### 7.2 Plots

TODO (leo): precision-recall curves (`fig_pr_curves.png`), model comparison (`fig_metrics_bars.png`),
threshold trade-off, calibration, confusion matrices, loss curves where a model has them. Each caption quotes the
figure's recreate command from `FIGURES.md`.

### 7.3 Important features

TODO (Jawad): permutation or model-based importance for the selected model, with the clinical sense-check.

### 7.4 Bias audit

TODO (Mashie): performance by hospital, age band and gender; report gaps with intervals; say what was not checked.
Evidence: `docs/hospital-a-vs-b.md`, `src/sepsis_warning/hospital_comparison.py`.

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
