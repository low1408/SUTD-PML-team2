# Team Task Tracker

## How to update

- Each owner updates their task rows as work progresses.
- Use **Not started**, **In progress**, **Blocked**, **In review**, **Done**, or **Deferred** for status.
- Set a due date and last-updated date using `YYYY-MM-DD`. A dash means not set.
- Add a PR, test result, or document link in **Evidence / notes**. For blocked tasks, name the blocker and next action.
- Mark a task **Done** only after its deliverable is verified and reviewed. Link private data evidence without uploading patient data.
- Statuses start as **Not started** until owners confirm existing work; they are not an assessment of current implementation.
- Share updates through commits or PRs once this project is in a shared Git repository. Markdown tables are manually maintained.

## Owners and slides

| Person | Owner | Slides |
|---|---|---|
| 1 | Edwin | 4, 5, 9, 10 |
| 2 | Jawad | 6, 7 |
| 3 | Kaiyuan | 3, 8, 11, 12 |
| 4 | Josh | 14 |
| 5 | leo | 16, 17, 18 |
| 6 | Mashie | 13, 15, 19, 20 |

Slide numbers follow the 22-slide deck in `docs/presentation/` (see its README for titles). Slides 1, 2, 21 and 22 (cover, problem, lessons, close) are shared and assembled by Mashie (M06). Slide 5 (bedside to model) sits with Edwin for now; Mashie may move it to Leo's serving story.

## Task status

### Person 1 - Edwin

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| E01 | Download data and convert to Parquet | Done | - | 2026-11-09 | Full data: 40,336 records / 1,552,210 rows; 0 SHA-256 mismatches vs manifest. `tests/test_prepare.py` (corrupt input, overwrite refusal, values kept exactly). Merged by Leo in PR #48; no formal Approve review. |
| E02 | Version data with DVC | In review | - | 2026-11-10 | `dvc.yaml`, `dvc.lock`, `docs/data-versioning.md`. Two private remotes, set outside Git: a folder on one laptop, and a private SUTD SharePoint site with no other members (4 prepared files, 20 MB, checksums match). No credentials in Git. Fresh clone restored 4 files with matching SHA-256. Licence question R14 open, so no teammates added yet; teammates can rebuild the same files from PhysioNet using the doc. Optional install; awaiting team OK for the package and remotes. |
| E03 | Data schema checks with Pandera | In review | - | 2026-11-09 | `validate` command and `schema.py`: all 40,336 stays / 1,552,210 rows pass the hard rules. Extreme values are counted for review on train only, never rejected; small counts hidden. `tests/test_schema.py` (16 tests). Optional install; awaiting team OK for the package. |
| E04 | Hospital A / B split | Done | - | 2026-11-09 | `split` command writes `splits.csv` (seed 42, hashes). 55/15/15/15 of A, all B external, every stay once, load-order independent. `tests/test_splits.py`. Stay-level, not person-level. Merged by Leo in PR #48; no formal Approve review. `baseline` can now read the saved split. |
| E05 | Exploratory analysis | In progress | - | 2026-11-09 | `compare-hospitals` command, 7 figures in `reports/figures/`, findings in `docs/hospital-a-vs-b.md`. A restricted to train split (no test data). To do: team OK on descriptive use of B. Hospital B wording on hold until the model owner replies. |
| E06 | SIRS rule baseline | Done | - | 2026-11-09 | `baseline` command → `baseline_metrics.csv`; SIRS F1 0.063 (A val), utility −0.014 at B. Independently recomputed with the official utility formula (match to 1e-15). Adds accuracy, specificity, confusion matrix and a validation-only threshold sweep (fig 8). `tests/test_baselines.py`. Merged by Leo in PR #48; no formal Approve review. Failure patterns: `docs/sirs-failure-patterns.md`. |
| E07 | Prepare slides 4, 5, 9, 10 | Not started | - | 2026-11-09 | Deck README now lists slides 4, 5, 9, 10; awaiting confirmation. Slides done last. |

### Person 2 - Jawad

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| J01 | Causal rolling features | Done | - | 2026-11-11 | Shared builder in `features.py`: last value, missing-now, never-seen, hours since seen, and a trailing window of mean, std, count and change. Hours since seen use ICULOS, including a stay that starts after hour 1 (`test_stay_starting_after_hour_one_uses_iculos`). Prefix and future-corruption tests in `tests/test_invariants.py`. Merged in PR #51, approved by Leo. `train` uses the 6-hour window. The 3-hour and 12-hour windows from `features_for_windows` are experimental and are not called by `train`. |
| J02 | Missing-value handling | Done | - | 2026-11-11 | Forward-fill stays inside one stay, has no expiry, and never fills an earlier blank from a later hour. `fit_training_fallback` and `apply_training_fallback` take medians from the tables they are given. Tests cover a validation row, an all-blank training column, an outage, unchanged raw rows, and a reset between patients. Merged in PR #53. `train` still fills blanks with sklearn's imputer. The training-only fallback is experimental and is not called by `train`. |
| J03 | Hours since last lab test | Done | - | 2026-11-11 | Per-lab hours since seen, missing-now and never-seen, measured in ICU hours from ICULOS. `test_lab_recency_covers_first_hour_gap_and_never_seen` and `test_stay_starting_after_hour_one_uses_iculos`. Merged in PR #51, approved by Leo. The late-start test is on this branch. |
| J04 | Leakage tests: corrupt hour t+1 and verify earlier features remain unchanged | Done | - | 2026-11-11 | Merged in PR #51: next-hour corruption, prefix scoring matches the full stay, saved-column mismatch refused. Merged in PR #53: training-only fit tests. `test_a_leaking_fill_changes_earlier_rows` substitutes a backward fill and shows the corruption check catches it. The shared builder on the same rows stays unchanged. |
| J05 | Fast utility-score function | Done | - | 2026-11-11 | `normalized_utility` matches the published example and is timed on 500 patients by 48 hours, 24,000 patient-hours, with no speed limit (`benchmark_utility`). Hourly decisions, separate from cooldown events. Merged in PR #51 and PR #53. |
| J06 | Shared feature module | Done | - | 2026-11-11 | `train`, replay and the scoring service call `build_features`. `scripts/causal_features.py` calls the same module. The saved column list is stored with the model and rejected when a stay cannot supply it. The exploratory `missingness` command still builds its own feature table. |
| J07 | Prepare slides 6, 7 | In progress | - | 2026-11-11 | Slides 6 and 7 are drafted in `docs/presentation/source/slides/pipeline.html` and `features.html`. Team review is not recorded. The features slide still lists 3, 6 and 12 hour windows as next work. The 3-hour and 12-hour windows exist as an experimental builder, and `train` uses 6 hours. |

### Person 3 - Kaiyuan

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| K01 | Logistic regression | In progress | - | 2026-11-13 | Implementing vectorized Logistic Regression from scratch in pure NumPy (no scikit-learn). Scikit-learn reference benchmark achieved AUROC 0.795, AP 0.105, Utility 0.250 (`artifacts/sklearn_benchmark/`). |
| K02 | Random forest | In progress | - | 2026-11-13 | Pure NumPy baseline implementation in progress. Scikit-learn reference benchmark achieved AUROC 0.817, AP 0.125, Utility 0.365 (`artifacts/sklearn_benchmark/`). |
| K03 | LightGBM as the main model | In progress | - | 2026-11-13 | Main model implementation from scratch in pure NumPy (no scikit-learn) in progress on branch `kaiyuan/model-from-scratch`. Scikit-learn HistGradientBoosting reference benchmark achieved AUROC 0.833, AP 0.136, Utility 0.390 (`artifacts/sklearn_benchmark/`). |
| K04 | Probability calibration | In progress | - | 2026-11-13 | Implementing pure NumPy Platt scaling (sigmoid calibration) on 15% calibration cohort with frozen base estimator. Scikit-learn reference benchmark achieved Brier score 0.0199. |
| K05 | Alarm threshold and six-hour cooldown | In progress | - | 2026-11-13 | Alert decision threshold tuning under <=0.20 false alarm ceiling with 6h cooldown in progress for NumPy models. Scikit-learn reference benchmark selected threshold 0.06 (0.192 false alarms/patient-day). |
| K06 | Fairness checks | In progress | - | 2026-11-13 | Subgroup audit across age groups (<50, 50-64, 65-79, >=80) and recorded gender in progress. |
| K07 | Prepare slides 3, 8, 11, 12 | In progress | - | 2026-11-13 | Preparing empirical evidence, model selection rationale, and cross-hospital generalization drop for slides 3, 8, 11, and 12. |

### Person 4 - Josh

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| H01 | HMM forward-pass states | Not started | - | - | - |
| H02 | Signature features | Not started | - | - | - |
| H03 | With / without tests | In review | - | 2026-11-14 | PR #71: `scripts/h03_ablation.py` runs arms A-D from [docs/h03-ablation-plan.md](docs/h03-ablation-plan.md) (train/calibration/validation only); metrics and paired bootstrap in `ablation.py`, 20 tests (`tests/test_ablation.py`). Real-data trial with `--limit 300` runs end to end. Full-cohort run pending: needs more than 8 GB RAM. |
| H04 | Optional: GMM | Not started | - | - | - |
| H05 | Optional: GRU | Not started | - | - | - |
| H06 | Prepare slide 14 | Not started | - | - | - |

### Person 5 - leo

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| L01 | MLflow tracking and registry | In review | - | 2026-11-14 | Local SQLite store (`sqlite:///artifacts/mlflow.db`; MLflow 3 refuses the old `file:` store). Selection runs: parent with fingerprints and the selected model's validation metrics, one nested run per candidate with validation metrics, settings, fit time and training-loss curve; no held-out test metrics are logged. Registry (`registry.py`): `train --register-model` registers the bundle as `@candidate` with its SHA-256 and data/split/feature hashes; `promote --reviewed-by` moves `@production` and records the previous version; `serve --model-uri models:/sepsis-warning@production` checks the hash before loading. `tests/test_registry.py` runs against real MLflow (CI installs the tracking extra). See `docs/mlflow.md`. |
| L02 | FastAPI service | In review | - | 2026-11-14 | `serve` command: `GET /health`, `GET /model` (selected model and validation-only candidate comparison, registry version), `POST /score`, `POST /predict` (up to 64 stays, latest-hour risk and alert); stateless full-history requests, 422 invalid input, 409 model mismatch, localhost-only by default, no auth ([service.py](src/sepsis_warning/service.py), [api.py](src/sepsis_warning/api.py)). Tests: replay parity, prefix scoring, concurrent isolation, error cases (`tests/test_service.py`, `tests/test_api.py`). Synthetic models only. FastAPI as optional `serve` extra needs team OK. |
| L03 | Docker image | In review | - | 2026-11-13 | `Dockerfile` (non-root uid 10001, pinned `requirements.txt`, model mounted read-only, allowlist `.dockerignore`, health check). `scripts/docker_smoke_test.sh` passes: non-root, no data/model files in the image, synthetic demo model, `/health`, `/score`, 422 on a label in the input. CI job `docker-image` runs it. See `docs/serving.md`. |
| L04 | CI/CD with GitHub Actions | Done | - | 2026-11-13 | [.github/workflows/tests.yml](.github/workflows/tests.yml): read-only permissions, no secrets, refuses committed data/model files, runs unit tests, the causal-feature script and the synthetic demo. No deployment. Green on main `5a0a5ea` (fork run #7, identical commit) and on `47896ad` (run #6). |
| L05 | Latency tests | In review | - | 2026-11-13 | `scripts/latency.py` (standard library): p50/p95/p99, throughput and failures per stay length and concurrency, synthetic stays. Docker image, 2 CPUs: p50 41 ms / p95 67 ms / p99 116 ms for a 24 h stay, 49/75/121 ms for 336 h, one client; requests queue under concurrency (about 15-20 req/s, single worker by design). No numeric target agreed yet. `docs/serving.md`. |
| L06 | AWS deployment design | In progress | - | 2026-11-13 | CloudFormation in `infra/cloudformation/`, static validation only (cfn-lint clean), no AWS deployment. Load balancer is now internal in private subnets by default, matching the deck's private design; `internet-facing` remains for a synthetic-only demo. README table lists what is aligned and what is not built (VPN, private API Gateway with OAuth2, SageMaker, EventBridge, Secrets Manager, CloudTrail). |
| L07 | Prepare slides 16, 17, 18 | Not started | - | - | - |

### Person 6 - Mashie

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| M01 | Hospital B replay | Not started | - | - | - |
| M02 | Drift monitoring at two speeds | In review | - | 2026-11-08 | `monitor-setup` / `monitor-replay` commands ([monitoring.py](src/sepsis_warning/monitoring.py)); plan, owners and runbook in [docs/monitoring-plan.md](docs/monitoring-plan.md). Reference from A train/validation only; thresholds frozen before any test replay. 18 tests incl. no-shift and simulated shifts (`tests/test_monitoring.py`). To do: team review of owners/floors; real-data reference after the K03-K05 freeze. |
| M03 | SHAP explanations | Not started | - | - | - |
| M04 | Model card | Not started | - | - | - |
| M05 | Risk register | In review | - | 2026-11-08 | [docs/risk-register.md](docs/risk-register.md): 25 risks with likelihood/impact, owner, evidence and remaining exposure; unresolved items left open. To do: each owner confirms their rows. |
| M06 | Final deck assembly | Not started | - | - | - |
| M07 | Prepare slides 13, 15, 19, 20 | Not started | - | - | - |

## Task details and acceptance criteria

These are proposed deliverables for owner review, not new implementation approvals.
Reuse existing code and tests before adding tools. New dependencies, storage
services, deployment infrastructure, and substantial experiments require approval.
Record verification evidence in the status tables above.

### Edwin - data and baselines

| ID | Scope / deliverable | Acceptance criteria |
|---|---|---|
| E01 | Maintain the official A/B download and combined Parquet preparation commands. Preserve source, record identity, original values, missingness, and labels. | Verify raw hashes and Parquet counts; for the full cohort expect 40,336 records and 1,552,210 hours. Test corrupt input, schema consistency, and overwrite refusal. Document reproducible commands. |
| E02 | Version the approved dataset and provenance with DVC, using an approved private remote. | Demonstrate an authorized fresh-checkout restore with matching hashes. Keep data, DVC cache, and credentials out of Git; commit only appropriate metadata. Confirm remote access and redistribution terms before uploading. |
| E03 | Define Pandera schema checks for required columns, numeric types, finite observed values, binary nondecreasing labels, and record-local consecutive hours. | Valid fixtures pass; missing columns, invalid labels, infinities, and broken chronology fail explicitly. Missing clinical values remain permitted. Separate extreme-value review flags from hard rejection rules. |
| E04 | Preserve source provenance and the existing record-level split: A train/calibration/validation/internal test; B external test. Persist seed, hashes, and assignments. | Repeated runs reproduce assignments, every record belongs to one split, and all B is external. Document approximate 55/15/15/15 A proportions and the lack of guaranteed person-disjointness. |
| E05 | Summarize training record/hour counts, label prevalence, record lengths, missingness, and value ranges. Use validation only for prespecified checks. | Produce reproducible aggregate tables/charts with units and denominators. Exclude test data from exploration and selection. Keep patient trajectories private and clear notebook outputs. |
| E06 | Implement a transparent SIRS-style rule comparator using available measurements; document thresholds, missing-component handling, and applicability. | Test threshold boundaries and missing inputs; prove prefix causality. Evaluate under the agreed label and alert metrics, and state where available data cannot reproduce full clinical SIRS criteria. Do not claim it diagnoses sepsis. |
| E07 | Prepare slides 4, 5, 9, 10 covering Edwin's agreed data and baseline contributions. | Slides link to verified evidence, identify data/split versions, distinguish completed from planned work, and contain no patient-level outputs. Obtain team review. |

### Jawad - features and metric implementation

| ID | Scope / deliverable | Acceptance criteria |
|---|---|---|
| J01 | Extend the shared causal feature builder with last-seen values and trailing-window mean, standard deviation, count, and change features. | Define window/hour semantics and behavior with short histories and missing measurements. Prefix and future-corruption tests pass; no operation crosses record boundaries. |
| J02 | Implement record-local forward fill and training-fitted fallback values; document any carry-forward expiry policy before changing it. | Leading missing values do not use future observations. Validation/test values never affect fitted statistics. Test entirely unobserved training columns, outages, and record resets; raw stored data stays unchanged. |
| J03 | Provide per-lab time since last observation, current-missingness, and never-seen indicators. | Test first hour, never observed, newly observed, and long gaps using ICU-hour semantics. Future appended measurements leave earlier features unchanged. |
| J04 | Strengthen leakage tests by corrupting hour t+1 onward and changing labels and excluded metadata. | Features through t remain identical; include scoring prefix checks for a fixed fitted model. Test record isolation and training-only preprocessing fits. Fail tests when an intentional leaking implementation is substituted. |
| J05 | Reuse or optimize the published benchmark utility implementation without changing its semantics. | Match published/reference examples and an independent comparison within a documented tolerance. Test no-positive and censored cases. Use hourly decisions, not cooldown events; benchmark runtime on a stated workload without inventing a speed target. |
| J06 | Keep feature creation shared across training, replay, and future serving. Expose a consistent feature order, dtype, and configuration contract. | The same history produces the same features in each consumer. Save the feature contract with model artifacts and test schema/version mismatches. Avoid duplicate notebook or API feature implementations. |
| J07 | Prepare slides 6 and 7: the end-to-end pipeline and causal feature engineering. | Show the feature contract, imputation fit boundary, and leakage-test evidence; clearly distinguish implemented features from proposals. Obtain team review. |

### Kaiyuan - models, calibration, and alarm policy

| ID | Scope / deliverable | Acceptance criteria |
|---|---|---|
| K01 | Establish a reproducible logistic-regression baseline using shared features, training-only imputation/scaling, and documented regularization/class weighting. | Compare against the constant-prior baseline on validation; report AP, AUROC, Brier, and convergence. Save fit settings and verify split isolation. Do not select from test results. |
| K02 | Evaluate a bounded random-forest baseline on the same feature/split contract. | Record seed, tree settings, resource usage, and validation metrics. Keep preprocessing training-fitted; distinguish feature importance from causal evidence. |
| K03 | Add LightGBM as the proposed main model after dependency/resource approval and compare it with the baselines. | Document missing-value and imbalance handling. Any tuning or early stopping uses explicitly designated record-disjoint development data, never an automatic patient-hour split or test data. Report validation results and do not assume it wins. |
| K04 | Fit probability calibration on the separate calibration cohort with the underlying estimator frozen. | No estimator refit on calibration records. Report before/after Brier and calibration curves on validation, including sample sizes and bin definitions. Save and reload the calibrated pipeline. |
| K05 | Select a validation-only threshold under the documented alert budget and apply a six-hour cooldown. | Test threshold equality, hourly timing, record resets, and no-feasible-threshold behavior. Save the chosen policy; report hourly utility and notification burden separately. Do not retune on B. |
| K06 | Audit performance by available age and recorded-gender groups using prespecified definitions. | Include subgroup record/hour counts, positive counts, calibration, detection, and alert burden with grouped uncertainty where feasible. Flag insufficient samples and label limitations; do not claim clinical fairness from this benchmark alone. |
| K07 | Prepare slides 3, 8, 11, 12 covering the agreed modeling contributions. | Include model comparisons, calibration/policy settings, and subgroup limitations with traceable evidence. Distinguish validation selection from final testing. Obtain team review. |

### Josh - temporal feature experiments

| ID | Scope / deliverable | Acceptance criteria |
|---|---|---|
| H01 | Fit an HMM on training data and expose normalized forward-filtered state probabilities from observations through the current hour. | Test prefix invariance, missing-observation handling, numerical stability, and state normalization. Do not use full-sequence smoothing or Viterbi states as online features; latent states are not established clinical stages. |
| H02 | Define causal path-signature features over approved trailing windows, with documented channels, time encoding, truncation depth, and missing-value policy. | Test prefix invariance, short-history behavior, deterministic feature order, and computational cost. Fit any preprocessing on training only and obtain dependency approval. |
| H03 | Run controlled with/without ablations for HMM and signature features. | Use the same splits, base feature contract, evaluation metrics, and declared tuning budget. Report validation performance and resource differences; defer final test comparison until choices are frozen. |
| H04 | Optional: evaluate training-fitted GMM memberships as an additional representation. | Approve scope first; test deterministic fitting, held-out transformation, and prefix-safe inputs. Compare against the baseline on validation; do not interpret components as validated sepsis subtypes. |
| H05 | Optional: prototype a GRU using causal histories with record-level sequence handling. | Approve dependency and compute scope first. Test padding/masking, hidden-state resets, and prefix scoring. Log training settings and resource use; use no future labels/observations as inputs. |
| H06 | Prepare slide 14 explaining temporal representations and ablations. | Include forward-only diagrams, with/without evidence, compute trade-offs, and limitations. Label deferred experiments explicitly. Obtain team review. |

### leo - reproducibility and serving

| ID | Scope / deliverable | Acceptance criteria |
|---|---|---|
| L01 | Introduce MLflow tracking and an approved private model registry for reproducible experiments. | Log split/data fingerprints, seeds, parameters, versions, metrics, and feature/policy contracts. Demonstrate authorized model retrieval and scoring parity. Never upload patient records or secrets; registry promotion requires review. |
| L02 | Build a local FastAPI research-scoring service using the saved model and shared feature code. | Define input/output schemas, history/state semantics, health checks, and explicit invalid-input/model-mismatch errors. Test batch/replay parity and concurrent record isolation. Do not trigger training or downloads per request; agree authentication before exposure. |
| L03 | Package the approved service as a reproducible Docker image. | Build and smoke-test with synthetic requests as a non-root user. Keep patient data, credentials, and local models out of image layers; document private runtime artifact loading and resource configuration. |
| L04 | Add reviewed GitHub Actions for focused tests and approved build checks; keep deployment separate until authorized. | Workflows run on synthetic fixtures, use least-privilege permissions, and never expose patient data or secrets. Use internal language setup actions first. Verify the configured checks actually execute; do not add unattended clinical deployment. |
| L05 | Measure end-to-end service latency for agreed request sizes and concurrency. | Report p50/p95/p99, throughput, failures, cold/warm behavior, hardware, and workload. Include feature computation and artifact loading where applicable. Agree a numeric target before treating it as a pass/fail gate. |
| L06 | Produce a modular, statically validated CloudFormation design; do not deploy it before project readiness and owner approval. | Separate networking, artifact/state, service, and monitoring templates. Document the public unauthenticated research-demo boundary, private task networking, encryption, data locations, monitoring, rollback, cost, and validation limits. Obtain owner review before provisioning or uploading anything. |
| L07 | Prepare slides 16, 17, 18 covering tracking, serving, and deployment design. | Clearly separate working local capabilities from proposed cloud architecture. Include reproducibility and latency evidence without credentials or patient data. Obtain team review. |

### Mashie - replay, monitoring, explanations, and reporting

| ID | Scope / deliverable | Acceptance criteria |
|---|---|---|
| M01 | Replay B records only after model, feature, calibration, and policy choices are frozen. Use the saved artifact and shared scorer. | Report cross-source metrics, alert burden, timing limitations, and record-grouped uncertainty. No B-guided retuning; keep record-level replay outputs private and distinguish research replay from live clinical use. |
| M02 | Design two-speed monitoring: rapid input/schema/missingness checks and slower aggregate score, calibration, performance, and alert-burden checks. | Define windows, reference data, thresholds, minimum samples, and response owners before B evaluation. Test simulated shifts; calibration/performance monitoring requires delayed labels, while unlabeled shifts do not prove performance loss. No automatic retraining from alerts. |
| M03 | Produce SHAP explanations for the fixed model using a training-derived background/reference. | Document explainer settings and supported model output; check explanation consistency and runtime. Use aggregate shareable summaries; keep patient explanations private. Explain associations, not causality or clinical recommendations. |
| M04 | Write a model card covering intended use, dataset/label semantics, splits, features, selection, calibration, metrics, and limitations. | Cite reproducible run evidence and subgroup results. Include privacy, missingness/shift sensitivity, timing/censoring, out-of-scope clinical use, and unresolved gaps. Obtain team review. |
| M05 | Maintain a risk register covering leakage, data quality, source shift, privacy, fairness, alert fatigue, licensing, and operational failures. | Each risk has likelihood/impact, owner, mitigation, evidence, and remaining exposure. Record unresolved risks instead of marking unsupported mitigations complete. |
| M06 | Assemble the final deck from the reviewed owner contributions. | Check slide order, terminology, units, denominators, citations, and agreement between results and artifacts. Remove patient-level outputs and unsupported clinical claims; complete a team walkthrough. |
| M07 | Prepare slides 13, 15, 19, 20 covering Mashie's agreed evaluation and reporting contributions. | Include frozen-model replay, monitoring design, explanations/risks, and final limitations as agreed with the team. Label synthetic and real-cohort evidence separately. Obtain team review. |

## Shared project boundaries

- Keep patient data under the Git-ignored `data/` directory. DVC metadata may be versioned; patient data must not be committed to Git.
- Preserve record-level split isolation. Use training data for fitting, calibration data for calibration, and validation data for model and threshold selection.
- Freeze model, feature, calibration, and alarm-policy decisions before Hospital B replay or held-out evaluation.
- Use only current and past observations for features; do not shift the supplied benchmark label again.
- Treat these assignments as planned work, not confirmation that the listed tools or capabilities are already implemented.
