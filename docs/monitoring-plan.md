# Two-speed monitoring plan (M02)

**Status:** In review. **Owner:** Mashie. **Date:** 2026-11-08.
**Code:** [src/sepsis_warning/monitoring.py](../src/sepsis_warning/monitoring.py),
[src/sepsis_warning/scoring.py](../src/sepsis_warning/scoring.py).
**Tests:** [tests/test_monitoring.py](../tests/test_monitoring.py).

This plan defines how the frozen early-warning model would be watched once it
scores a new population, and it provides a research replay that runs the same
checks on held-out PhysioNet records. It is a research design. It does not
approve live clinical use, and nothing in it retrains or retunes the model.

## 1. Why two speeds

Sepsis labels arrive late: onset is confirmed only after cultures and
antibiotics are charted, often after the stay ends. Inputs and scores are
available every hour. So the plan has two speeds:

| Speed | Runs on | Needs labels? | Answers |
|---|---|---|---|
| **Fast** | every fast window (default 2,000 patient-hours) | No | Do the inputs and scores still look like hospital A? Is the alert rate stable? |
| **Slow** | every slow window of stays with settled labels (default 20,000 patient-hours) | Yes | Is the model still calibrated, discriminating and useful? Are false alerts and subgroup gaps acceptable? |

**A fast alarm is a prompt to investigate, not proof that performance
dropped.** Inputs can shift without hurting the model, and the model can
degrade without a visible input shift. Only the slow checks speak to
performance, and they need settled labels.

## 2. What is checked

All statistics are computed per window; larger means worse.

| Check | Speed | Statistic | Reference | Minimum sample | Owner |
|---|---|---|---|---|---|
| `schema` | fast | Missing required columns + non-numeric + infinite values + non-consecutive hours within a stay | Required PhysioNet columns | none | Edwin |
| `missingness` | fast | Absolute change in the share of missing hours, per variable (per stay for Age, Gender, HospAdmTime) | A train | none | Jawad |
| `input_drift` | fast | PSI of observed values, 10 reference-quantile bins, per variable; Age, Gender, HospAdmTime compared one value per stay | A train | 200 observed hours (30 stays for the per-stay inputs) | Jawad |
| `new_variable` | fast | Number of variables never observed in A train that now carry values (for example EtCO2, which A never records) | A train | none | Jawad |
| `score_drift` | fast | PSI of hourly probabilities | A validation, frozen bundle | none | Kaiyuan |
| `alert_rate` | fast | \|ln((alerts per patient-day + 0.01) / (reference + 0.01))\|, all stays | A validation | none | Kaiyuan |
| `calibration` | slow | Change in ln(observed positive-hour rate / mean probability) | A validation | 300 stays and 20 septic stays per window | Kaiyuan |
| `discrimination` | slow | Hourly AUROC drop | A validation | as above | Mashie |
| `utility` | slow | Normalized hourly utility drop (published PhysioNet weights, hourly decisions) | A validation | as above | Mashie |
| `early_detection` | slow | Early-detection recall drop (alert 0 to 12 h before inferred onset; left-censored stays excluded) | A validation | as above | Mashie |
| `alert_burden` | slow | Increase in ln(false alerts per non-septic patient-day) | A validation | as above | Kaiyuan |
| `subgroup_gap` | slow | Largest best-minus-worst utility within recorded sex or within age band (<45, 45-64, 65-79, 80+) | none (absolute) | 10 septic stays per group, two groups per attribute | Kaiyuan |

Brier score, average precision and mean probability are reported alongside but
are **not** alarm statistics, because they move with prevalence (B has 5.7%
septic stays against A's 8.8%), even when the model is unchanged.

## 3. Reference data

| What | Reference | Why |
|---|---|---|
| Input distributions and missingness | Hospital A **train** split (11,183 stays) | The data the model learned from. |
| Score distribution, alert rate, performance | Hospital A **validation** split scored by the **frozen bundle** | Scores must come from data the estimator was not fitted on; validation is the cohort the project already uses for selection, so using it here adds no new held-out exposure. |
| No-shift variability for thresholds | Windows resampled from A validation | Tells how much each statistic moves by chance at the chosen window size. |

The internal test split and hospital B are **never read** while building the
reference. `reference_from_cohorts` only accesses the `train` and `validation`
cohorts, and a test passes it a dictionary holding only those two, so any other
access would fail. The reference therefore depends only on A and the frozen
bundle, whenever it is built.

Reference performance is slightly optimistic, because the alert threshold was
chosen on validation. The resampled thresholds absorb part of this; the rest
is a stated limitation.

## 4. Windows

PhysioNet records have no calendar time. The replay therefore simulates a
stream: whole stays are shuffled with a fixed seed (42) and cut into
consecutive windows of at least *N* patient-hours. A stay is never split
across windows. A final window shorter than *N* is not evaluated, and its size
is reported.

| Setting | Default | Reasoning |
|---|---:|---|
| Fast window | 2,000 patient-hours | About one day of hourly rows from an 80-bed ICU; at the A median stay length (39 h) roughly 50 stays. |
| Slow window | 20,000 patient-hours | Roughly 520 stays: about 45 septic stays at A's rate and 30 at B's, above the 20-septic minimum. |
| Minimum per slow window | 300 stays, 20 septic stays | Fewer septic stays make AUROC, utility and early recall too noisy to act on. |
| Minimum per subgroup | 10 septic stays | Smaller groups are listed as insufficient, never scored. |
| Fast persistence | 2 windows | A fast alarm needs two consecutive breaches; the first breach is a warning (schema errors alarm at once). |
| No-shift resamples | 200 fast and 200 slow windows | Enough for a stable 99th percentile. |

In a live service the same windows would be time-based (fast hourly over the
last day; slow weekly over stays whose labels settled). The patient-hour sizes
above were chosen to approximate that.

Settings are frozen into `monitoring_reference.json`. A replay always uses the
reference's settings, never new command-line values. A smaller local sample
(for example 500 records per source) needs smaller windows; `monitor-setup`
stops with a clear message when A validation is too small for a window.

## 5. Thresholds

For every statistic:

```text
warn  = max(warn floor,  95th percentile on no-shift A-validation windows)
alert = max(alert floor, 99th percentile on no-shift A-validation windows, warn)
```

The floors stop a very stable statistic from alarming on trivial changes. The
percentiles stop a noisy statistic from alarming on chance variation at the
chosen window size. Both are fixed before any held-out replay.

| Check | Warn floor | Alert floor | Basis |
|---|---:|---:|---|
| `missingness` | 5 points | 10 points | On the full cohort B differs from A (train) by 34 points on DBP and 8 to 12 points on Resp, FiO2, BaseExcess, Glucose and pH ([missingness_by_hospital.csv](../reports/figures/missingness_by_hospital.csv)). |
| `input_drift`, `score_drift` | PSI 0.10 | PSI 0.20 | Common PSI convention (0.1 moderate, 0.25 major); 0.2 matches the deck's monitoring slide. |
| `alert_rate` | x1.5 | x2 | Alert volume is what staff experience. |
| `calibration` | O/E x1.25 | O/E x1.5 | Calibration-in-the-large. |
| `discrimination` | AUROC -0.03 | AUROC -0.05 | |
| `utility` | -0.05 | -0.10 | Normalized utility. |
| `early_detection` | -0.10 | -0.20 | About 230 eligible septic stays give roughly +/-6.5 points of noise ([design page](design/sepsis-warning-pipeline.html)). |
| `alert_burden` | x1.5 | x2 | False alerts per non-septic patient-day; the validation budget is 0.2. |
| `subgroup_gap` | 0.10 | 0.20 | Utility units, absolute. |
| `schema` | 1 error | 1 error | Fixed; any error alarms immediately. |
| `new_variable` | 1 variable | never alarms | Fixed; the model has never seen these values, so the owner checks how its features treat them. |

About 80 statistics are checked per fast window (two per input variable). At a
99th-percentile threshold, single-window breaches are expected by chance. That
is why fast alarms need two consecutive windows, and why a lone warning is only
logged.

## 6. Owners and responses

Owners come from [TEAM-TASKS.md](../TEAM-TASKS.md) and must be confirmed in
team review.

| Owner | Role | Checks |
|---|---|---|
| Edwin | Data and schema (E01, E03, E04) | `schema` |
| Jawad | Features and missing values (J01-J06) | `missingness`, `input_drift`, `new_variable` |
| Kaiyuan | Model, calibration, alert policy, fairness (K03-K06) | `score_drift`, `alert_rate`, `calibration`, `alert_burden`, `subgroup_gap` |
| Mashie | Evaluation and monitoring (M01, M02) | `discrimination`, `utility`, `early_detection`; maintains this plan and the runbook |

| Status | Response |
|---|---|
| `ok` | No action. |
| `warn` | Log it; review at the weekly team check. |
| `alert` | Owner reviews within one working day using the runbook below. |
| `insufficient_data` | Too little data in the window. Never read it as ok. |

### Runbook for an alert

1. **Confirm.** Re-run the replay and check the window's counts. Rule out a
   pipeline fault first: schema errors, a renamed field, or a unit change (for
   example temperature charted in Fahrenheit) explain many input alarms.
2. **Triage.**
   - Schema: fix the data feed; scores from broken windows are not trusted.
   - Missingness, input drift or a new variable: find the charting or
     ordering change. Ask whether features built on "was it measured?" still
     mean the same thing.
   - Score drift or alert rate: check the fast input checks first; then wait
     for the slow checks before claiming a performance change.
   - Calibration, discrimination, utility, early detection, burden: compare
     with the fast checks and with the subgroup table.
   - Subgroup gap: hand to the fairness owner (K06) with group sizes.
3. **Decide and record** in the GitHub issue, with the evidence: no action,
   fix the data feed, or propose model work.
4. **No automatic retraining.** Retraining, recalibration or a threshold
   change needs team approval. It produces a new frozen bundle and a new
   monitoring reference, and it must be evaluated on data not used to make the
   change. Hospital B stays a test set: retraining on B would remove the only
   cross-hospital estimate.

## 7. Label settlement

Slow checks use only stays whose labels have settled. A window with any missing
label is reported as `insufficient_data` (tested). In the retrospective replay
every stay is complete. A live service would admit a stay to the slow checks
only after discharge plus the label-confirmation delay; that delay must be
agreed with clinical partners and is not set here.

## 8. How to run

Synthetic smoke run (software check only; the synthetic B has a +4 bpm HR offset
and 10 points more missingness, which the monitor should flag):

```bash
python -m sepsis_warning demo
python -m sepsis_warning monitor-setup --data data/synthetic --model artifacts/demo/model.joblib \
  --output artifacts/monitoring/demo-reference --fast-window-hours 300 --slow-window-hours 1000 \
  --min-observed-values 30 --min-observed-stays 5 --min-slow-stays 15 --min-slow-septic-stays 3 \
  --min-subgroup-septic-stays 2
python -m sepsis_warning monitor-replay --data data/synthetic --model artifacts/demo/model.joblib \
  --reference artifacts/monitoring/demo-reference/monitoring_reference.json \
  --split external_test --output artifacts/monitoring/demo-replay-b --after-freeze
```

`monitor-replay` refuses `internal_test` and `external_test` unless `--after-freeze`
is given, which records that the model has been frozen. It also refuses a
bundle whose SHA-256 differs from the reference.

Real data, once the team has frozen the model, features, calibration and alert
policy:

```bash
python -m sepsis_warning monitor-setup --data data/physionet --model artifacts/real/model.joblib \
  --output artifacts/monitoring/reference
python -m sepsis_warning monitor-replay --data data/physionet --model artifacts/real/model.joblib \
  --reference artifacts/monitoring/reference/monitoring_reference.json \
  --split external_test --output artifacts/monitoring/replay-b --after-freeze
```

## 8a. Hospital B replay (M01)

The external-test replay adds a cross-source comparison to the window checks.
It is written to `cross_source.json` and a section of `monitoring_report.md`:

- **Same metrics on both sources**: hospital A validation (the frozen model's
  own reference population, scored by the same bundle) and hospital B, for
  normalized utility, AUROC, average precision, false alerts per non-septic
  patient-day and early-detection recall.
- **Record-grouped uncertainty**: a bootstrap over whole stays, 200 draws, seed
  42. Hours within a stay are never treated as independent. Each source is
  resampled on its own; the B-minus-A interval uses the paired draws.
- **Timing limitations**: counts of stays, septic stays, left-censored septic
  stays, septic stays ending before the inferred onset, and stays eligible for
  early detection. Recall is always reported with its denominator.

The replay applies the frozen threshold and cooldown. Nothing on hospital B
changes a threshold, a cooldown or the reference. The synthetic fixture is
used only to test the code: its numbers say nothing about the real hospital B.

On the full cohort, building the slow no-shift thresholds takes roughly two
minutes after the records load (about 0.5 s per slow window).

| Output | Contents |
|---|---|
| `monitoring_reference.json` | Settings, reference distributions, thresholds, owners, responses, bundle SHA-256 |
| `fast_windows.csv`, `slow_windows.csv` | One row per window: counts, statistics, status per statistic |
| `slow_subgroups.csv` | Per window and group: stays, septic stays, utility, false-alert burden |
| `summary.json`, `monitoring_report.md` | Counts per check, most-drifted inputs, alert rate and performance against the reference, pooled subgroups |
| `monitoring_overview.png` | Input drift, alert rate, utility and calibration per window |

All outputs are aggregates; no record identifiers are written (tested). They
still belong under the git-ignored `artifacts/` directory until reviewed. Only
reviewed aggregate figures go into `reports/` or the deck.

Safeguards in the commands:

- The replay refuses a model whose SHA-256 differs from the reference, a split
  other than `internal_test` or `external_test`, a non-empty output folder, and
  a data mode (synthetic or PhysioNet) that differs from the reference.
- Setup refuses a bundle whose synthetic flag disagrees with the data, and
  window sizes that A validation cannot fill.

## 9. Simulated shifts tested

| Scenario | Expected | Test |
|---|---|---|
| No shift (A internal test) | No alarm in any fast or slow window | `test_no_alarm_on_unshifted_source_hospital_stays` |
| Temperature charted in Fahrenheit | `input_drift` warns in the first window and alarms in the next | `test_unit_change_is_detected_after_two_windows` |
| HR missing in 60% of hours | `missingness` alarm | `test_missingness_shift_is_detected` |
| Column dropped; one hour missing | `schema` alarms immediately, only in the affected window | `test_schema_breaks_alert_immediately` |
| EtCO2 starts being recorded | `new_variable` warns, never alarms | `test_variable_never_seen_in_training_only_warns` |
| Scores rise, ranking unchanged | `score_drift` and `calibration` alarm; `discrimination` does not | `test_score_and_calibration_shift_without_ranking_change` |
| Scores shuffled (ranking lost) | `discrimination` and `utility` alarm | `test_lost_ranking_is_a_slow_performance_alarm` |
| Labels not yet available | Every slow check `insufficient_data`; fast checks still run | `test_slow_checks_need_settled_labels` |
| Synthetic hospital B through the CLI | `missingness` and HR `input_drift` alarms; aggregate outputs only | `test_cli_freezes_reference_then_replays_hospital_b` |

## 10. Relation to other work

- **M01 (hospital B replay)** uses the same scorer, `scoring.score_patients`.
  The cross-source numbers belong to M01; this plan supplies the monitoring
  view of the same replay (deck slide "Monitoring the hospital shift").
- **K06 (fairness)** owns the subgroup audit. Monitoring only tracks the gap
  over time and flags small groups.
- **L05/L06**: the CloudWatch alarms in
  [infra/cloudformation/monitoring.yaml](../infra/cloudformation/monitoring.yaml)
  cover service health (CPU, memory, 5xx, latency, healthy targets) and do not
  monitor the model. In the target AWS design, these checks would run as
  scheduled jobs (hourly fast, weekly slow) and alarm the owners above.

## 11. Limitations

- Research replay of retrospective records; not live monitoring.
- The stream is simulated, so it shows the step change from A to B, not
  gradual drift within a hospital over time.
- Thresholds are calibrated on A validation, which also chose the alert
  threshold; reference performance is slightly optimistic.
- Calibration-in-the-large can miss miscalibration that cancels out across
  the probability range; reliability curves are K04's.
- Subgroups are limited to recorded sex and age band; there is no person ID,
  and no other attributes are available.
- The real-data reference has not been built yet: it needs the frozen bundle
  from K03-K05.

## 12. Acceptance criteria (issue #39)

| Criterion | Evidence |
|---|---|
| Fast input/schema/missingness checks and slower score, calibration, performance and alert-burden checks | Sections 1-2; `fast_statistics`, `slow_statistics` |
| Windows, reference data, thresholds, minimum samples and response owners defined before B evaluation | Sections 3-6; frozen into `monitoring_reference.json`; the reference never reads test splits (tested) |
| Simulated shifts tested | Section 9; 18 tests in `tests/test_monitoring.py` |
| Calibration and performance monitoring need delayed labels | Section 7; `test_slow_checks_need_settled_labels` |
| Unlabeled shifts do not prove performance loss | Section 1; stated in every report and summary |
| No automatic retraining from alerts | Section 6 runbook; `automatic_retraining: false` in the reference and every summary; the module has no training code |
| Hospital B replay with the cross-source comparison and record-grouped uncertainty (M01) | Section 8a; `cross_source.py`; `tests/test_hospital_b_replay.py` (9 tests, including the freeze gate and a whole-stay bootstrap) |
| Still open | Team review of owners and floors; the real-data reference and the real hospital B replay after the model freeze (both need the team's freeze decision) |
