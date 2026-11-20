# Risk register (M05)

**Maintainer:** Mashie. **Last reviewed:** 2026-11-19 (second review, against
`main` at `ee6ca9f`: after the NumPy models, the MLOps phases 1-4 and the M01
replay tooling merged; no model has been frozen yet). **Status:** In review;
owners must confirm their rows.

This register lists what could make the project's results wrong, unsafe to
share, or unusable, and what is actually in place against each risk. A
mitigation counts as **in place** only when it has evidence (code, a test, a
document or an output). Anything planned but not done stays listed under
**remaining exposure**, and the risk stays open.

## Scales

| Score | Likelihood (before the end of the project or in a first deployment) | Impact |
|---:|---|---|
| 1 | Unlikely | Small rework; no result or claim changes |
| 2 | Possible | Invalidates one result, figure or slide; days of rework |
| 3 | Likely, or already observed | Invalidates a core claim, breaches privacy or licence terms, or puts patients at risk if deployed |

**Rating = likelihood x impact:** 1-2 low, 3-4 medium, 6-9 high.

**Status:** *Open* (no effective mitigation yet), *Partly mitigated* (some
controls with evidence, exposure remains), *Mitigated* (controls with
evidence; residual exposure accepted by the owner), *Accepted* (known and
deliberately not mitigated, with a reason). A row marked *Mitigated (owner to
confirm)* has the evidence, but its owner has not yet accepted the residual
exposure.

Review the register at each milestone: before any hospital B evaluation,
before the deck freeze, and before submission (6 Dec 2026).

## What changed in this review

- **Lower:** R01, R05, R10, R18, R19 and R22, because the tests, checks or
  decisions they were waiting for are now on `main` (details below).
- **Status improved, same rating:** R03, R11, R16, R17.
- **Higher:** R21. The `deploy` workflow ran against an AWS account on
  2026-11-19 (leo's fork), so "nothing is deployed" no longer holds.
- **New:** R26 (a drift alarm can start retraining), R27 (contribution
  records disagree), R28 (the written report has no owner).
- **New exposure on existing rows:** R03 (several commands can read hospital B
  labels), R13 (the client falls back to SIRS), R17 (CI is red on `main`), R20
  (live drift checks use their own thresholds), R23 (LightGBM wording).

## Summary

| ID | Category | Risk | L | I | Rating | Owner | Status | Change |
|---|---|---|---:|---:|---:|---|---|---|
| R01 | Leakage | Future observations leak into features or scores | 1 | 3 | 3 Medium | Jawad | Mitigated (owner to confirm) | lower |
| R02 | Leakage | Same stay or person on both sides of a split | 2 | 2 | 4 Medium | Edwin | Partly mitigated | same |
| R03 | Leakage | Held-out outcomes influence choices; repeated looks at hospital B | 2 | 3 | 6 High | Kaiyuan, Mashie | Partly mitigated | status |
| R04 | Leakage | Challenge label misused or re-shifted; censored onsets | 2 | 2 | 4 Medium | Mashie | Partly mitigated | same |
| R05 | Data quality | Invalid or implausible raw values | 1 | 2 | 2 Low | Edwin | Partly mitigated | lower |
| R06 | Data quality | Heavy missingness and imputation choices bias the model | 3 | 2 | 6 High | Jawad | Partly mitigated | same |
| R07 | Data quality | Synthetic, subset or untraceable results reported as evidence | 2 | 3 | 6 High | Kaiyuan, Mashie | Partly mitigated | same |
| R08 | Source shift | Model does not transfer from hospital A to B | 3 | 3 | 9 High | Mashie | Partly mitigated | same |
| R09 | Source shift | Upstream feed changes in deployment (units, fields, outages) | 2 | 2 | 4 Medium | Edwin, Mashie | Partly mitigated | same |
| R10 | Privacy | Patient data or patient-level outputs leave the local machine | 1 | 3 | 3 Medium | Edwin (all) | Partly mitigated | lower |
| R11 | Privacy | Scoring API reachable without proper authentication | 1 | 3 | 3 Medium | leo | Partly mitigated | status |
| R12 | Fairness | Unequal performance by sex or age; groups too small to judge | 2 | 3 | 6 High | Kaiyuan | Open | same |
| R13 | Alert fatigue | Too many false alerts; budget not clinically set | 3 | 2 | 6 High | Kaiyuan | Partly mitigated | same |
| R14 | Licensing | Dataset licence notices disagree (CC BY 4.0 vs ODbL) | 2 | 2 | 4 Medium | Edwin | Open | same |
| R15 | Licensing | Third-party icons and cited figures in the deck | 1 | 1 | 1 Low | Mashie | Partly mitigated | same |
| R16 | Operational | Loading an untrusted or wrong model file | 1 | 3 | 3 Medium | leo | Mitigated (owner to confirm) | status |
| R17 | Operational | Results cannot be reproduced; quality gate red | 2 | 2 | 4 Medium | leo | Partly mitigated | status |
| R18 | Operational | Full-cohort runs exhaust memory or time | 1 | 2 | 2 Low | Jawad, Kaiyuan | Mitigated (owner to confirm) | lower |
| R19 | Operational | Duplicate feature and metric code drifts (training/serving skew) | 1 | 2 | 2 Low | Jawad | Mitigated (owner to confirm) | lower |
| R20 | Operational | Monitoring raises false alarms, misses drift, or is inconsistent | 2 | 2 | 4 Medium | Mashie, leo | Partly mitigated | same |
| R21 | Operational | Cloud cost and retained resources | 2 | 2 | 4 Medium | leo | Open | higher |
| R22 | Compliance | Work breaks the course rules (no deep learning) | 1 | 3 | 3 Medium | Josh, team | Mitigated | lower |
| R23 | Communication | Deck or report makes unsupported or inconsistent claims | 3 | 2 | 6 High | Mashie | Open | same |
| R24 | Clinical misuse | Scores read as a diagnosis or a validated six-hour risk | 1 | 3 | 3 Medium | Mashie | Partly mitigated | same |
| R25 | Delivery | No frozen model yet; dependent evaluation runs late | 3 | 2 | 6 High | Mashie, team | Open | same |
| R26 | Operational | A drift alarm can start retraining automatically | 2 | 2 | 4 Medium | leo, Mashie | Open | new |
| R27 | Communication | Contribution and status records disagree or misattribute work | 3 | 2 | 6 High | leo, each owner | Open | new |
| R28 | Delivery | The written report (30% of the grade) has no owner or draft | 3 | 3 | 9 High | Team | Open | new |

## Details

### R01. Future observations leak into features or scores (Leakage)

- **Risk:** a feature at hour *t* uses hour *t+1* or later (backfill, centred
  windows, whole-stay summaries, smoothing), so validation results are
  inflated and do not hold in real time.
- **In place:** forward-fill only, trailing windows
  ([features.py](../src/sepsis_warning/features.py)); prefix, future-corruption
  and prefix-scoring tests, plus a deliberately leaking backward fill that the
  check must catch (`test_a_leaking_fill_changes_earlier_rows`) in
  [test_invariants.py](../tests/test_invariants.py) (J04, done); the HMM forward
  filter and the signature features each have a "later hours do not change
  earlier ones" test ([test_hmm.py](../tests/test_hmm.py),
  [test_hmm_fit.py](../tests/test_hmm_fit.py),
  [test_signature.py](../tests/test_signature.py)); `scripts/causal_features.py`
  now calls the shared builder instead of copying it.
- **Remaining exposure:** the experimental 3-hour and 12-hour windows
  (`features_for_windows`) are not used by `train`; any feature set added later
  must pass the same prefix tests before it enters a comparison.

### R02. Same stay or person on both sides of a split (Leakage)

- **Risk:** rows from one stay, or stays from one person, appear in training
  and testing, inflating results.
- **In place:** whole stays assigned to one split, seeded and stratified, with
  disjointness and load-order tests ([splits.py](../src/sepsis_warning/splits.py),
  [test_splits.py](../tests/test_splits.py)); `splits.csv` with SHA-256 hashes;
  `train`, `baseline` and `evaluate` read the saved split, and `evaluate`
  refuses a split whose hash differs from the one the model was trained on
  ([evaluation.py](../src/sepsis_warning/evaluation.py)).
- **Remaining exposure:** the data has no person ID, so one person could have
  stays in two splits. This cannot be fixed with this dataset; it is reported
  as a limitation.

### R03. Held-out outcomes influence choices; repeated looks at hospital B (Leakage)

- **Risk:** test results (A internal test or B) are seen before choices are
  final, or seen several times, and then used to adjust features, models or
  thresholds, so the "external" estimate is no longer independent.
- **In place:** `train` no longer scores the test splits; it selects on
  validation only (PR #77). The `evaluate` command scores the test splits once
  per frozen model and refuses a second look on real data, recorded as a
  registry tag or a `frozen_evaluation.json` marker
  ([evaluation.py](../src/sepsis_warning/evaluation.py)). `monitor-replay` and
  `ward` refuse the test splits unless `--after-freeze` is given
  ([ward.py](../src/sepsis_warning/ward.py) `check_split`). The monitoring
  reference reads only A train and validation (tested). Promotion gates use
  validation numbers only ([gates.py](../src/sepsis_warning/gates.py)).
- **Remaining exposure:**
  - Several commands can still read hospital B labels: `evaluate`,
    `monitor-replay --split external_test --after-freeze`,
    `ward --after-freeze`, and `baseline` (SIRS on B, already reported as
    utility -0.014). Only `evaluate` records the look and blocks a repeat.
  - `--after-freeze` is a flag the person running the command sets; nothing
    checks it against a recorded freeze decision or the `evaluate` marker.
  - B inputs and aggregate outcomes were already explored
    ([data-profile.md](data-profile.md), [hospital-a-vs-b.md](hospital-a-vs-b.md)),
    so B must be reported as exploratory external evidence, not an untouched
    holdout.
- **Next action:** record the freeze (bundle SHA-256 and date) in the tracker
  before any B run; run hospital B once, through one command, and report
  every look that happened.

### R04. Challenge label misused or re-shifted; censored onsets (Leakage)

- **Risk:** `SepsisLabel` is already shifted six hours early and stays positive
  after onset. Shifting it again, or describing scores as a clinical
  six-hour risk ("sepsis in the next 6 hours"), misstates the result. Stays
  positive from their first row have unknown onset (12.1% of A-train septic
  stays, 19.5% of B).
- **In place:** the label is never shifted in code; the utility uses the
  published weights; utility-weighted training targets use training labels
  only; left-censored stays are excluded from lead time and counted in the
  M01 timing table ([cross_source.py](../src/sepsis_warning/cross_source.py));
  documented in [README.md](../README.md) and [data-profile.md](data-profile.md).
- **Remaining exposure:** the cover slide still says the score "warns up to 6
  hours before sepsis onset" (see R23). The written model card (M04) does not
  exist yet.

### R05. Invalid or implausible raw values (Data quality)

- **Risk:** corrupt files, wrong types or impossible values enter training.
- **In place:** `read_patient`/`validate_record` and `prepare` reject missing
  columns, non-numeric and infinite values, non-binary or decreasing labels
  and broken hour sequences ([test_prepare.py](../tests/test_prepare.py));
  Pandera rules and the `validate` command pass all 40,336 stays / 1,552,210
  rows, and implausible readings are counted for review on train only, never
  rejected ([schema.py](../src/sepsis_warning/schema.py),
  [test_schema.py](../tests/test_schema.py), E03); the scoring service rejects
  values outside hard physical limits (422); the monitoring `schema` check
  repeats the core rules on new data.
- **Remaining exposure:** E03 is in review and Pandera is an optional extra
  awaiting team OK; the counted extremes have not been reviewed by anyone with
  clinical knowledge.

### R06. Heavy missingness and imputation choices bias the model (Data quality)

- **Risk:** most labs are missing in 82-99% of hours. Carrying values forward
  without a limit, or learning "was it measured?", can encode care practice
  rather than physiology.
- **In place:** missingness, never-seen and hours-since-seen features;
  training-only median fill (scikit-learn imputer in `train`; the J02
  training-only fallback is implemented and tested but not called by `train`);
  the `missingness` audit command; the monitoring `missingness` check.
- **Remaining exposure:** carry-forward has no expiry, and no decision is
  recorded (the [MLOps audit](mlops-audit.md) suggests a staleness-limit
  ablation and a note in the model card); no with/without-missingness-features
  ablation (H03 tested HMM and signature features only).

### R07. Synthetic, subset or untraceable results reported as evidence (Data quality)

- **Risk:** numbers from the synthetic fixture, the 500-record sample, or a run
  nobody can reproduce end up in the report or deck as cohort results.
- **In place:** `SYNTHETIC.json` marker and banners; data mode written to every
  output and manifest; commands refuse mismatched data modes; the
  `real_data` promotion gate (`promote --require-real-data`); the H03 slide
  states its validation-only scope.
- **Remaining exposure:** the real-data validation numbers circulating for the
  NumPy and scikit-learn models come from a closed, unmerged PR (#72) and from
  `artifacts/sklearn_benchmark/`, which is not in the repository; no
  registered real-data model or manifest is recorded; the deck still has
  placeholders that must be filled only from recorded runs.

### R08. Model does not transfer from hospital A to B (Source shift)

- **Risk:** B differs from A: 5.7% vs 8.8% septic stays, very different
  missingness (DBP 34 points; Resp, FiO2 and several labs 8-12 points), higher
  MAP, EtCO2 recorded only at B, and more left-censored septic stays.
  Calibration, alert burden and utility can all degrade. The SIRS baseline
  already drops from utility 0.084 (A validation) to -0.014 (B).
- **In place:** B held out from all fitting and selection; descriptive
  comparison ([hospital-a-vs-b.md](hospital-a-vs-b.md)); two-speed monitoring
  with thresholds frozen from A only ([monitoring-plan.md](monitoring-plan.md));
  the M01 cross-source report with a whole-stay bootstrap and timing counts
  ([cross_source.py](../src/sepsis_warning/cross_source.py),
  [test_hospital_b_replay.py](../tests/test_hospital_b_replay.py)); rule:
  report degradation, do not retune on B.
- **Remaining exposure:** no model result at B yet (waits for the freeze, R25);
  no domain adaptation is planned; a single B estimate from one public cohort.

### R09. Upstream feed changes in deployment (Source shift)

- **Risk:** a renamed field, unit change (for example temperature in
  Fahrenheit), new charting habit or feed outage silently changes features.
- **In place:** monitoring `schema`, `missingness`, `input_drift` and
  `new_variable` checks with tests for a dropped column, a missing hour, a
  Fahrenheit unit change and a newly recorded variable
  ([test_monitoring.py](../tests/test_monitoring.py)); the API rejects unknown
  fields, labels, gaps and physically impossible values (422); the live drift
  job tracks missingness per measurement
  ([live_drift.py](../src/sepsis_warning/live_drift.py)).
- **Remaining exposure:** no live feed exists; the variable and unit mapping
  document for a hospital integration (design page) is not written.

### R10. Patient data or patient-level outputs leave the local machine (Privacy)

- **Risk:** raw records, Parquet, predictions, replay HTML, notebook outputs or
  model files are committed, uploaded or synchronised.
- **In place:** `.gitignore` covers `data/`, `artifacts/`, `*.psv`,
  `*.joblib`, `*.pkl`, credentials and local chat exports; CI fails if data or
  model files are committed ([tests.yml](../.github/workflows/tests.yml));
  committed notebooks have no saved outputs; committed figures and CSVs are
  aggregates; monitoring and M01 outputs are aggregate-only (tested); the
  prediction log stores a hashed stay ID and no measurement values
  ([telemetry.py](../src/sepsis_warning/telemetry.py)); the DVC remote is a
  private SharePoint site with no other members
  ([data-versioning.md](data-versioning.md)).
- **Remaining exposure:** `.gitignore` does not stop cloud sync of a checkout
  (iCloud, OneDrive); `train` writes patient-level `validation_predictions.csv`
  and `replay.html`, and `evaluate` writes `held_out_predictions.csv` and
  `replay.html`, all locally; they must not be shared.

### R11. Scoring API reachable without proper authentication (Privacy)

- **Risk:** an endpoint that accepts patient data is reachable by others.
- **In place:** the local API binds to localhost and refuses other hosts
  unless `--allow-non-local-host` is passed; the AWS load balancer is internal
  by default; the Lambda function URL uses IAM authentication
  ([serverless.yaml](../infra/cloudformation/serverless.yaml)); deployment is
  started by hand only and does nothing until an OIDC deploy role is set as a
  repository variable ([deploy.yml](../.github/workflows/deploy.yml)).
- **Remaining exposure:** no user-level authentication (OAuth2 is a design,
  not built); the `deploy` workflow ran in leo's fork on 2026-11-19 and its
  "Stack and images" step succeeded (model publish and serving were skipped),
  so what is reachable in that account should be confirmed. Synthetic
  requests only.

### R12. Unequal performance by sex or age (Fairness)

- **Risk:** the model works worse for some groups; B is younger (median 62 vs
  65) with fewer patients aged 80+. Only recorded sex and age are available.
- **In place:** subgroup utility and false-alert burden by recorded sex and
  age band (<45, 45-64, 65-79, 80+) in the slow monitoring checks, with groups
  under 10 septic stays marked insufficient
  ([monitoring.py](../src/sepsis_warning/monitoring.py)).
- **Remaining exposure:** no fairness report exists for any real model (K06 is
  in progress on `main`, "In review" only in the open PR #80); other
  attributes cannot be assessed with this data.

### R13. Too many false alerts (Alert fatigue)

- **Risk:** staff ignore a tool that alarms too often. SIRS raises 1.98 false
  alerts per non-septic patient-day, ten times the 0.2 budget.
- **In place:** validation-only threshold under the budget with a six-hour
  cooldown; hourly decisions counted separately from notifications (tests in
  [test_invariants.py](../tests/test_invariants.py)); the `within_alert_budget`
  promotion gate; monitoring tracks alert rate (fast) and false-alert burden
  (slow).
- **Remaining exposure:** the 0.2 budget is a demonstration value, not set by
  clinicians; the budget can be exceeded at B; when the API is unavailable the
  reference client falls back to the SIRS rule
  ([client.py](../src/sepsis_warning/client.py)), which on validation raises
  about ten times the budget, so an outage would flood alerts unless the ward
  side suppresses fallback alerts.

### R14. Dataset licence notices disagree (Licensing)

- **Risk:** the PhysioNet project page states CC BY 4.0, while the dataset's
  own `LICENSE.txt` (fetched from
  `physionet.org/files/challenge-2019/1.0.0/LICENSE.txt` on 2026-11-08) is
  the ODC Open Database License, which adds share-alike terms for derived
  databases. The project docs cite only CC BY 4.0.
- **In place:** no data or derived database is redistributed; the DVC remote
  has no members besides its owner until this is settled
  ([data-versioning.md](data-versioning.md)); the dataset is cited.
- **Remaining exposure:** unresolved. It blocks sharing prepared data with
  teammates, so each member must rebuild from PhysioNet. Ask PhysioNet or the
  course instructor.

### R15. Third-party icons and cited figures in the deck (Licensing)

- **Risk:** AWS Architecture Icons and the statistics on the problem slide
  (Kumar 2006, Seymour 2017) are used without checking terms or wording.
- **In place:** attribution in the [presentation README](presentation/README.md)
  and on the slides.
- **Remaining exposure:** icon terms and quoted figures to be checked during
  the deck assembly (M06); the new hourly-data infographic
  (`docs/design/hourly-data-to-prediction.html`) also shows the AWS
  deployment.

### R16. Loading an untrusted or wrong model file (Operational)

- **Risk:** Joblib files can execute code when loaded, and the wrong bundle
  gives wrong scores.
- **In place:** `load_bundle` checks the SHA-256 recorded in `manifest.json`
  (or an explicit hash) before opening the file, and `replay`, `serve`,
  `evaluate` and `monitor-replay` all use it; registry versions carry the
  bundle hash; the service refuses a bundle built with another scikit-learn
  major.minor ([manifest.py](../src/sepsis_warning/manifest.py),
  [scoring.py](../src/sepsis_warning/scoring.py)).
- **Remaining exposure:** a bundle without `manifest.json` (from before PR #77)
  still loads without a check unless a hash is given.

### R17. Results cannot be reproduced; quality gate red (Operational)

- **Risk:** a teammate or marker cannot rebuild the same results, or a broken
  change reaches `main` unnoticed.
- **In place:** pinned requirements with hashes in `requirements/` and
  `uv.lock` (the old `poetry.lock` is gone); seeds, package versions, split and
  data hashes and a feature-contract hash in every run's manifest; CI runs
  lint, type checks, the tests, the causal-feature script and the synthetic
  demo on every PR ([tests.yml](../.github/workflows/tests.yml)).
- **Remaining exposure:**
  - CI on `main` has failed since 2026-11-17 at the "Known vulnerabilities in
    the locked dependencies" step (`pip-audit`), so the gate is red.
  - On macOS, `test_artifacts_stay_next_to_the_database` in
    [test_registry.py](../tests/test_registry.py) fails because `/var` and
    `/private/var` are compared unresolved (322 of 323 tests pass,
    2026-11-19); Linux CI is not affected.
  - [testing/real-data.md](testing/real-data.md) still lists
    `held_out_predictions.csv` as a `train` output, which `train` no longer
    writes.

### R18. Full-cohort runs exhaust memory or time (Operational)

- **Risk:** the full cohort (1.55 million hours) is processed in memory.
- **In place:** Parquet preparation writes in batches; scoring runs in
  batches of 1,000 stays; full validation runs have completed on teammates'
  machines (H03 on all 3,051 validation stays; the NumPy model check in
  PR #72, about 20 minutes).
- **Remaining exposure:** `train` still builds feature tables in memory; no
  memory figure is recorded for a full training run.

### R19. Duplicate feature and metric code drifts (Operational)

- **Risk:** features and utility are implemented in several places, so
  training and serving disagree.
- **In place:** `train`, `replay`, the API and `scripts/causal_features.py`
  call `build_features` (J06); the saved column list is stored with the model
  and refused when a stay cannot supply it; API scores are tested against
  batch replay (`test_scores_match_batch_replay` in
  [test_service.py](../tests/test_service.py)); the manifest stores a hash of
  the feature and policy code.
- **Remaining exposure:** the exploratory `missingness` command keeps its own
  feature table (not used for any model).

### R20. Monitoring raises false alarms, misses drift, or is inconsistent (Operational)

- **Risk:** about 80 statistics per window cause alarm fatigue for the team;
  real drift is missed; two monitoring paths disagree.
- **In place:** thresholds are the larger of a floor and the no-shift
  99th percentile; fast alarms need two consecutive windows; no alarm on the
  unshifted A test split and correct alarms on simulated shifts in the tests;
  named owners and a runbook ([monitoring-plan.md](monitoring-plan.md)).
- **Remaining exposure:**
  - The live drift job (`monitor-live`,
    [live_drift.py](../src/sepsis_warning/live_drift.py)) uses its own fixed
    thresholds (score PSI 0.10/0.25, alert-rate ratio 0.5-2x and 0.25-4x,
    missingness 0.10/0.20) instead of the frozen thresholds in
    `monitoring_reference.json`, and the monitoring plan does not describe it.
  - The real-data reference is not built; owners and floors await team
    review; the stream is simulated, so gradual drift is not exercised.

### R21. Cloud cost and retained resources (Operational)

- **Risk:** a deployment incurs charges; retained S3, KMS, ECR and DynamoDB
  resources stay billable after teardown.
- **In place:** optional budget alert; a cost estimate in
  [aws-serverless.md](aws-serverless.md); deployment is manual-only.
- **Remaining exposure:** the `deploy` workflow ran in leo's fork on
  2026-11-19 and the "Stack and images" step succeeded, so a stack and
  container images probably exist in that account. The design now includes
  SageMaker, an MLflow App, Lambda and VPC endpoints. Budget alerts do not cap
  spend, and retained resources survive a stack deletion. Owner to confirm
  what exists, the budget, and the teardown plan.

### R22. Work breaks the course rules (Compliance)

- **Risk:** the course brief does not allow deep learning models.
- **In place:** H05 (GRU) is deferred in [TEAM-TASKS.md](../TEAM-TASKS.md)
  ("Deep learning is outside the course rules"); all models are classical
  (logistic regression, random forest, gradient boosting, HMM and signature
  features); sequence models appear only as future work on the lessons slide.
- **Remaining exposure:** none known; keep "future work" wording clear.

### R23. Deck or report makes unsupported or inconsistent claims (Communication)

- **Risk:** slides state results that were not run, or contradict each other
  and the repository.
- **In place:** placeholders are marked; the design page and deck label cloud
  work as a design; slide numbers in TEAM-TASKS and the deck README now agree
  (22-slide deck); the Docker demo the closing slide mentions now exists (L03).
- **Remaining exposure (checked 2026-11-19):**
  - Slides 12, 13 and 19 still contain `[__]` placeholders, and slides 15 and
    20 their bracketed prompts.
  - The cover says the score "warns up to 6 hours before sepsis onset" (R04).
  - The lessons slide still mentions an insurance and claims-triage use case
    unrelated to this project.
  - The ladder slide says "LightGBM next", but K03 is a NumPy
    histogram gradient-boosting model, not LightGBM.
  - GitHub issue titles still use the old slide numbers (for example #44
    "slides 10, 12, 15, 16"; the deck assigns 13, 15, 19, 20).
  - The `.pptx` has not been rebuilt since slide 14 changed.

### R24. Scores read as a diagnosis or a validated six-hour risk (Clinical misuse)

- **Risk:** a reader treats the research score as clinical decision support.
- **In place:** the [clinical-use plan](clinical-use-design-and-validation-plan.md)
  sets release gates; replay, monitoring and API outputs carry research-only
  wording; `GET /model` serves a validation-only selection card and never test
  results ([model_card.py](../src/sepsis_warning/model_card.py)).
- **Remaining exposure:** the written model card (M04) does not exist yet;
  cover-slide wording (R04).

### R25. No frozen model yet; dependent evaluation runs late (Delivery)

- **Risk:** hospital B replay (M01), the real monitoring reference (M02),
  explanations (M03), the model card's numbers (M04), the fairness report
  (K06) and the results slides all wait for one frozen, registered model;
  submission is due 6 Dec 2026.
- **In place:** the tooling for each is built and tested on synthetic data
  (`evaluate`, `monitor-setup`, `monitor-replay`, `cross_source.py`, gates,
  registry).
- **Remaining exposure:** as of 2026-11-19 no model is frozen: K01-K06 are
  "In progress" on `main` and "In review" only in the open PR #80; no
  real-data model is registered. Agree the candidate, the run, the reviewer
  and a freeze date at the next team check.

### R26. A drift alarm can start retraining automatically (Operational)

- **Risk:** M02 requires "no automatic retraining from alerts". In
  [observability.yaml](../infra/cloudformation/observability.yaml), setting
  `RetrainImageTag` makes a drift alarm start the training pipeline
  (`"trigger": "drift-alarm"`); the pipeline still stops at a named approver
  before release ([drift-alarm runbook](runbooks/drift-alarm.md)).
- **In place:** the option is off by default (empty `RetrainImageTag`); release
  needs an approver; the offline monitoring code has no training path.
- **Remaining exposure:** if enabled, training starts without a person
  deciding to retrain, which conflicts with the M02 plan. Decide as a team:
  remove the option, or document it in the monitoring plan as an exception
  with its approval step.

### R27. Contribution and status records disagree or misattribute work (Communication)

- **Risk:** the course requires each member's contribution in the submission.
  Inconsistent records misstate who did what.
- **In place:** owners update [TEAM-TASKS.md](../TEAM-TASKS.md); a mistaken
  author mention in PR #72 was corrected in its comments.
- **Remaining exposure (checked 2026-11-19):**
  - [contributions.md](../contributions.md) marks M02 and M05 "Done" while
    TEAM-TASKS marks them "In review"; K01-K06 differ between the two files.
  - PR #80, which syncs them, is open with merge conflicts.
  - Work done by one member under another's task (for example M01 tooling in
    PR #81) needs to be credited correctly.
- **Next action:** one source of truth (TEAM-TASKS), merged before the
  contributions section of the report is written.

### R28. The written report has no owner or draft (Delivery)

- **Risk:** the brief requires a 10-15 page PDF report (30% of the grade):
  executive summary, problem, data, methods, evaluation with a bias audit,
  failure examples, limitations, run instructions, figure-recreation steps,
  a pipeline diagram and member contributions. No task in
  [TEAM-TASKS.md](../TEAM-TASKS.md) owns it, and no draft is in the repository.
- **In place:** much of the source material exists (README, design page,
  monitoring plan, this register, hospital comparison, H03 plan, SIRS failure
  patterns).
- **Remaining exposure:** unassigned work with 17 days to submission. Assign
  an owner and sections at the next team check.

## Change log

| Date | Change | By |
|---|---|---|
| 2026-11-08 | First version: 25 risks from a full repository review | Mashie |
| 2026-11-19 | Second review against `main` at `ee6ca9f`: six risks lowered, R21 raised, four statuses improved, R26-R28 added, new exposure recorded on R03, R13, R17, R20, R23 | Mashie |
