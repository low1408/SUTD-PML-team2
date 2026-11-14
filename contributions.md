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
| K01 | Logistic regression | Not started | - | - | - |
| K02 | Random forest | Not started | - | - | - |
| K03 | LightGBM as the main model | Not started | - | - | - |
| K04 | Probability calibration | Not started | - | - | - |
| K05 | Alarm threshold and six-hour cooldown | Not started | - | - | - |
| K06 | Fairness checks | Not started | - | - | - |
| K07 | Prepare slides 3, 8, 11, 12 | Not started | - | - | - |

### Person 4 - Josh

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| H01 | HMM forward-pass states | Not started | - | - | - |
| H02 | Signature features | Not started | - | - | - |
| H03 | With / without tests | Not started | - | - | - |
| H04 | Optional: GMM | Not started | - | - | - |
| H05 | Optional: GRU | Not started | - | - | - |
| H06 | Prepare slide 14 | Not started | - | - | - |

### Person 5 - leo

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| L01 | MLflow tracking and registry | In review | - | 2026-11-14 | Local SQLite store (`sqlite:///artifacts/mlflow.db`; MLflow 3 refuses the old `file:` store). Selection runs: parent with fingerprints and the selected model's validation metrics, one nested run per candidate with validation metrics, settings, fit time and training-loss curve; no held-out test metrics are logged. Registry (`registry.py`): `train --register-model` registers the bundle as `@candidate` with its SHA-256 and data/split/feature hashes; `promote --reviewed-by` moves `@production` and records the previous version; `serve --model-uri models:/sepsis-warning@production` checks the hash before loading. `tests/test_registry.py` runs against real MLflow (CI installs the tracking extra). See `docs/mlflow.md`. |
| L02 | FastAPI service | Not started | - | - | - |
| L03 | Docker image | In review | - | 2026-11-13 | `Dockerfile` (non-root uid 10001, pinned `requirements.txt`, model mounted read-only, allowlist `.dockerignore`, health check). `scripts/docker_smoke_test.sh` passes: non-root, no data/model files in the image, synthetic demo model, `/health`, `/score`, 422 on a label in the input. CI job `docker-image` runs it. See `docs/serving.md`. |
| L04 | CI/CD with GitHub Actions | Done | - | 2026-11-13 | [.github/workflows/tests.yml](.github/workflows/tests.yml): read-only permissions, no secrets, refuses committed data/model files, runs unit tests, the causal-feature script and the synthetic demo. No deployment. Green on main `5a0a5ea` (fork run #7, identical commit) and on `47896ad` (run #6). |
| L05 | Latency tests | In review | - | 2026-11-13 | `scripts/latency.py` (standard library): p50/p95/p99, throughput and failures per stay length and concurrency, synthetic stays. Docker image, 2 CPUs: p50 41 ms / p95 67 ms / p99 116 ms for a 24 h stay, 49/75/121 ms for 336 h, one client; requests queue under concurrency (about 15-20 req/s, single worker by design). No numeric target agreed yet. `docs/serving.md`. |
| L06 | AWS deployment design | In progress | - | 2026-11-13 | CloudFormation in `infra/cloudformation/`, static validation only (cfn-lint clean), no AWS deployment. Load balancer is now internal in private subnets by default, matching the deck's private design; `internet-facing` remains for a synthetic-only demo. README table lists what is aligned and what is not built (VPN, private API Gateway with OAuth2, SageMaker, EventBridge, Secrets Manager, CloudTrail). |
| L07 | Prepare slides 16, 17, 18 | Not started | - | - | - |

### Person 6 - Mashie

| ID | Task | Status | Due date | Updated | Evidence / notes |
|---|---|---|---|---|---|
| M01 | Hospital B replay | Not started | - | - | - |
| M02 | Drift monitoring at two speeds | Done | - | 2026-11-10 | `monitor-setup` / `monitor-replay` ([monitoring.py](src/sepsis_warning/monitoring.py)); reference from A train/validation only, no automatic retraining. 18 tests in `tests/test_monitoring.py` (simulated shifts, label settlement, test-split isolation). Plan and runbook: [docs/monitoring-plan.md](docs/monitoring-plan.md). Merged in PR #52, approved by Leo. |
| M03 | SHAP explanations | Not started | - | - | - |
| M04 | Model card | Not started | - | - | - |
| M05 | Risk register | Done | - | 2026-11-10 | [docs/risk-register.md](docs/risk-register.md): 25 risks with likelihood, impact, owner, evidence and remaining exposure; unresolved risks left open. Merged in PR #52, approved by Leo. Owners still confirm their own rows (e.g. R05 once E03 merges). |
| M06 | Final deck assembly | Not started | - | - | - |
| M07 | Prepare slides 13, 15, 19, 20 | Not started | - | - | - |

## Shared project boundaries

- Keep patient data under the Git-ignored `data/` directory. DVC metadata may be versioned; patient data must not be committed to Git.
- Preserve record-level split isolation. Use training data for fitting, calibration data for calibration, and validation data for model and threshold selection.
- Freeze model, feature, calibration, and alarm-policy decisions before Hospital B replay or held-out evaluation.
- Use only current and past observations for features; do not shift the supplied benchmark label again.
- Treat these assignments as planned work, not confirmation that the listed tools or capabilities are already implemented.
