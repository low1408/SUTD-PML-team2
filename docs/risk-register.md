# Risk register (M05)

**Maintainer:** Mashie. **Last reviewed:** 2026-11-08 (first version, before
any real-data model run). **Status:** In review; owners must confirm their rows.

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
deliberately not mitigated, with a reason).

Review the register at each milestone: before any hospital B evaluation,
before the deck freeze, and before submission (6 Dec 2026).

## Summary

| ID | Category | Risk | L | I | Rating | Owner | Status |
|---|---|---|---:|---:|---:|---|---|
| R01 | Leakage | Future observations leak into features or scores | 2 | 3 | 6 High | Jawad | Partly mitigated |
| R02 | Leakage | Same stay or person on both sides of a split | 2 | 2 | 4 Medium | Edwin | Partly mitigated |
| R03 | Leakage | Held-out outcomes influence choices (B "untouched" no longer true) | 2 | 3 | 6 High | Kaiyuan, Mashie | Open |
| R04 | Leakage | Challenge label misused or re-shifted; censored onsets | 2 | 2 | 4 Medium | Mashie | Partly mitigated |
| R05 | Data quality | Invalid or implausible raw values | 2 | 2 | 4 Medium | Edwin | Partly mitigated |
| R06 | Data quality | Heavy missingness and imputation choices bias the model | 3 | 2 | 6 High | Jawad | Partly mitigated |
| R07 | Data quality | Synthetic or subset results reported as real-cohort evidence | 2 | 3 | 6 High | Kaiyuan, Mashie | Partly mitigated |
| R08 | Source shift | Model does not transfer from hospital A to B | 3 | 3 | 9 High | Mashie | Partly mitigated |
| R09 | Source shift | Upstream feed changes in deployment (units, fields, outages) | 2 | 2 | 4 Medium | Edwin, Mashie | Partly mitigated |
| R10 | Privacy | Patient data or patient-level outputs leave the local machine | 2 | 3 | 6 High | Edwin (all) | Partly mitigated |
| R11 | Privacy | Public, unauthenticated scoring API if deployed | 1 | 3 | 3 Medium | leo | Accepted while undeployed |
| R12 | Fairness | Unequal performance by sex or age; groups too small to judge | 2 | 3 | 6 High | Kaiyuan | Open |
| R13 | Alert fatigue | Too many false alerts; budget not clinically set | 3 | 2 | 6 High | Kaiyuan | Partly mitigated |
| R14 | Licensing | Dataset licence notices disagree (CC BY 4.0 vs ODbL) | 1 | 2 | 2 Low | Edwin | Accepted |
| R15 | Licensing | Third-party icons and cited figures in the deck | 1 | 1 | 1 Low | Mashie | Partly mitigated |
| R16 | Operational | Loading an untrusted or wrong model file | 1 | 3 | 3 Medium | leo | Partly mitigated |
| R17 | Operational | Results cannot be reproduced (dependencies, no CI) | 2 | 2 | 4 Medium | leo | Open |
| R18 | Operational | Full-cohort runs exhaust memory or time | 2 | 2 | 4 Medium | Jawad, Kaiyuan | Partly mitigated |
| R19 | Operational | Duplicate feature and metric code drifts (training/serving skew) | 2 | 2 | 4 Medium | Jawad | Open |
| R20 | Operational | Monitoring raises false alarms, misses drift, or triggers retraining | 2 | 2 | 4 Medium | Mashie | Partly mitigated |
| R21 | Operational | Cloud cost and retained resources | 1 | 2 | 2 Low | leo | Accepted while undeployed |
| R22 | Compliance | Work breaks the course rules (no deep learning) | 2 | 3 | 6 High | Josh, team | Open |
| R23 | Communication | Deck or report makes unsupported or inconsistent claims | 3 | 2 | 6 High | Mashie | Open |
| R24 | Clinical misuse | Scores read as a diagnosis or a validated six-hour risk | 1 | 3 | 3 Medium | Mashie | Partly mitigated |
| R25 | Delivery | Blocked dependencies delay evaluation before the deadline | 3 | 2 | 6 High | Mashie, team | Open |

## Details

### R01. Future observations leak into features or scores (Leakage)

- **Risk:** a feature at hour *t* uses hour *t+1* or later (backfill, centred
  windows, whole-stay summaries, smoothing), so validation results are
  inflated and do not hold in real time.
- **In place:** forward-fill only, trailing windows
  ([features.py](../src/sepsis_warning/features.py)); prefix and
  future-corruption tests (`test_feature_prefix_matches_streaming_history`,
  `test_future_changes_do_not_change_past_features`,
  `test_missing_value_is_not_backfilled` in
  [test_invariants.py](../tests/test_invariants.py)); causal SIRS tests in
  [test_baselines.py](../tests/test_baselines.py).
- **Remaining exposure:** J04's stronger tests (fixed-model score prefixes,
  intentionally leaking implementation must fail) are not done. Planned HMM,
  signature and multi-window features (H01, H02) are not covered yet. A
  second implementation in `scripts/causal_features.py` is not under the
  package tests.
- **Next action:** J04 tests; every new feature set must pass the prefix tests
  before it enters a comparison.

### R02. Same stay or person on both sides of a split (Leakage)

- **Risk:** rows from one stay, or stays from one person, appear in training
  and testing, inflating results.
- **In place:** whole stays assigned to one split, seeded and stratified, with
  disjointness and load-order tests ([splits.py](../src/sepsis_warning/splits.py),
  [test_splits.py](../tests/test_splits.py), `test_split_is_disjoint_and_b_is_external`);
  `splits.csv` with SHA-256 hashes.
- **Remaining exposure:** the data has no person ID, so one person could have
  stays in two splits. This cannot be fixed with this dataset; it is reported
  as a limitation.

### R03. Held-out outcomes influence choices (Leakage)

- **Risk:** test results (A internal test or B) are seen before choices are
  final and then used to adjust features, models or thresholds, so the
  "external" estimate is no longer independent.
- **In place:** selection code reads validation only
  ([cli.py](../src/sepsis_warning/cli.py) `run_experiment`, `tune_threshold`);
  the monitoring reference reads only A train and validation (tested in
  [test_monitoring.py](../tests/test_monitoring.py)); A-vs-B exploration uses
  A train only ([hospital-a-vs-b.md](hospital-a-vs-b.md)).
- **Remaining exposure:** the freeze is a team rule, not enforced in code:
  `train` selects the model and immediately scores both test splits in the
  same run, so re-running after seeing B is easy. B inputs and aggregate
  outcomes have already been inspected
  ([data-profile.md](data-profile.md), [hospital-a-vs-b.md](hospital-a-vs-b.md),
  the data-check notebook's source-shift table), so B must be reported as
  exploratory external evidence, not an untouched holdout
  ([architecture review](architecture-review-and-design.md)).
- **Next action:** record the frozen bundle hash before M01; describe B
  accurately in M01, M04 and the deck; the planned separate evaluate step
  (design page) would enforce the freeze.

### R04. Challenge label misused or re-shifted; censored onsets (Leakage)

- **Risk:** `SepsisLabel` is already shifted six hours early and stays positive
  after onset. Shifting it again, or describing scores as a clinical
  six-hour risk, misstates the result. Stays positive from their first row
  have unknown onset (12.1% of A-train septic stays, 19.5% of B).
- **In place:** the label is never shifted in code; the utility uses the
  published weights (`test_utility_matches_official_published_example`);
  left-censored stays are excluded from lead time
  (`test_lead_time_excludes_left_censored_records`); documented in
  [README.md](../README.md) and [data-profile.md](data-profile.md).
- **Remaining exposure:** the cover slide says the score "warns up to 6 hours
  before sepsis onset", which reads as a clinical claim (see R23). The model
  card (M04) is not written.

### R05. Invalid or implausible raw values (Data quality)

- **Risk:** corrupt files, wrong types or impossible values enter training.
- **In place:** `read_patient` and `prepare` reject missing columns,
  non-numeric and infinite values, non-binary or decreasing labels and broken
  hour sequences; output is never overwritten
  ([test_prepare.py](../tests/test_prepare.py)); full-cohort download checked
  against SHA-256 (E01: 40,336 records, 1,552,210 rows, no mismatches). The
  monitoring `schema` check repeats these checks on new data.
- **Remaining exposure:** no plausibility review of extremes (for example
  glucose 912 mg/dL, AST 9,840 IU/L in the sample); Pandera checks (E03) not
  done.

### R06. Heavy missingness and imputation choices bias the model (Data quality)

- **Risk:** most labs are missing in 82-99% of hours. Carrying values forward
  without a limit, or learning "was it measured?", can encode care practice
  rather than physiology.
- **In place:** missingness, never-seen and hours-since-seen features;
  training-only median fill; the `missingness` audit command
  ([missingness.py](../src/sepsis_warning/missingness.py)); the monitoring
  `missingness` check.
- **Remaining exposure:** no decision on a carry-forward expiry (J02); no
  with/without-missingness-features ablation yet (README item 4).

### R07. Synthetic or subset results reported as real evidence (Data quality)

- **Risk:** numbers from the synthetic fixture or the 500-record sample end up
  in the report or deck as cohort results.
- **In place:** `SYNTHETIC.json` marker; synthetic banner in the replay; data
  mode written to every output; commands refuse synthetic filenames without
  the marker, and monitoring refuses a mismatched data mode.
- **Remaining exposure:** no full-cohort model run is recorded yet; the deck
  still has `[__]` placeholders that must be filled only from recorded runs.

### R08. Model does not transfer from hospital A to B (Source shift)

- **Risk:** B differs from A: 5.7% vs 8.8% septic stays, very different
  missingness (DBP 34 points; Resp, FiO2 and several labs 8-12 points), higher MAP, EtCO2
  recorded only at B, and more left-censored septic stays. Calibration,
  alert burden and utility can all degrade. The SIRS baseline already drops
  from utility 0.084 (A validation) to -0.014 (B).
- **In place:** B held out from all fitting and selection; descriptive
  comparison ([hospital-a-vs-b.md](hospital-a-vs-b.md)); two-speed monitoring
  with thresholds frozen from A only, tested on simulated shifts
  ([monitoring-plan.md](monitoring-plan.md)); rule: report degradation, do not
  retune on B.
- **Remaining exposure:** no model result at B yet (M01 waits for K03-K05);
  no domain adaptation is planned; a single B estimate from one public cohort.

### R09. Upstream feed changes in deployment (Source shift)

- **Risk:** a renamed field, unit change (for example temperature in
  Fahrenheit), new charting habit or feed outage silently changes features.
- **In place:** monitoring `schema`, `missingness`, `input_drift` and
  `new_variable` checks, with tests for a dropped column, a missing hour, a
  Fahrenheit unit change and a newly recorded variable
  ([test_monitoring.py](../tests/test_monitoring.py)).
- **Remaining exposure:** no live feed exists; the variable and unit mapping
  document for a hospital integration (design page) is not written.

### R10. Patient data or patient-level outputs leave the local machine (Privacy)

- **Risk:** raw records, Parquet, predictions, replay HTML, notebook outputs or
  model files are committed, uploaded or synchronised.
- **In place:** `.gitignore` covers `data/`, `artifacts/`, `*.psv`,
  `*.joblib`, `*.pkl` and credentials; committed notebooks have no saved
  outputs (checked 2026-11-08); committed figures and CSVs are aggregates;
  monitoring writes aggregates only and a test checks that no record ID
  appears in its outputs; the course brief (contains a shared cluster
  password) is git-ignored.
- **Remaining exposure:** `.gitignore` does not stop cloud sync (for example
  Desktop or Documents synced to iCloud or OneDrive); each member must check
  where their checkout lives. Patient-level replay HTML from `train` is
  written locally and must not be shared.

### R11. Public, unauthenticated scoring API if deployed (Privacy)

- **Risk:** the CloudFormation prototype defaults to an internet-facing load
  balancer, open source range and HTTP without authentication.
- **In place:** nothing is deployed; the service is off by default
  (`DeployScoringService=false`); the README restricts any deployment to
  synthetic requests ([infra README](../infra/cloudformation/README.md)).
- **Remaining exposure:** authentication, TLS and a narrow source range are not
  implemented. Accepted only while undeployed.

### R12. Unequal performance by sex or age (Fairness)

- **Risk:** the model works worse for some groups; B is younger (median 62 vs
  65) with fewer patients aged 80+. Only recorded sex and age are available.
- **In place:** age bands shared between the comparison and monitoring; the
  slow monitoring check tracks the utility gap and marks groups with fewer
  than 10 septic stays as insufficient.
- **Remaining exposure:** the fairness audit (K06) has not started; no
  subgroup results exist; other attributes cannot be assessed with this data.

### R13. Too many false alerts (Alert fatigue)

- **Risk:** staff ignore a tool that alarms too often. SIRS raises 1.98 false
  alerts per non-septic patient-day, ten times the 0.2 budget.
- **In place:** validation-only threshold under the budget with a six-hour
  cooldown; hourly decisions counted separately from notifications (tests in
  [test_invariants.py](../tests/test_invariants.py)); monitoring tracks alert
  rate (fast) and false-alert burden (slow).
- **Remaining exposure:** the 0.2 budget is a demonstration value, not set by
  clinicians; the budget can be exceeded at B; no model result yet.

### R14. Dataset licence notices disagree (Licensing)

- **Risk:** the PhysioNet project page states CC BY 4.0, while the dataset's
  own `LICENSE.txt` (fetched from
  `physionet.org/files/challenge-2019/1.0.0/LICENSE.txt` on 2026-11-08) is
  the ODC Open Database License, which adds share-alike terms for derived
  databases. The project docs cite only CC BY 4.0.
- **In place:** the data and cleaned files are shared only inside the team, through
  the private SUTD SharePoint site (decided by Edwin, 2026-11-22). Both licences allow
  this with credit, and ODbL's share-alike terms apply to public use, which this is not.
  Nothing is posted publicly; the dataset is cited (Reyna et al.) in the README, data
  profile and deck.
- **Remaining exposure:** any public release of derived data (for example S3 in L06
  made public) would still need the licence question answered first.

### R15. Third-party icons and cited figures in the deck (Licensing)

- **Risk:** AWS Architecture Icons and the statistics on the problem slide
  (Kumar 2006, Seymour 2017) are used without checking terms or wording.
- **In place:** attribution in the [presentation README](presentation/README.md)
  and on the slides.
- **Remaining exposure:** icon terms and quoted figures to be checked during
  the deck assembly (M06).

### R16. Loading an untrusted or wrong model file (Operational)

- **Risk:** Joblib files can execute code when loaded, and the wrong bundle
  gives wrong scores.
- **In place:** only locally produced bundles are loaded; the monitoring replay
  checks the bundle's SHA-256 against the frozen reference and refuses a
  mismatch ([scoring.py](../src/sepsis_warning/scoring.py), tested).
- **Remaining exposure:** the `replay` command and the planned API do not check
  a hash yet.

### R17. Results cannot be reproduced (Operational)

- **Risk:** a teammate or marker cannot rebuild the same results.
- **In place:** seeds, package versions, split manifest and data hashes are
  saved with each run; the README documents commands.
- **Remaining exposure:** dependencies are version ranges in
  `pyproject.toml`; a `poetry.lock` exists but the project builds with
  setuptools, the lock includes a package not declared in `pyproject.toml`
  (`cloudpickle`), and Agent.md says no lockfile exists, so its status is
  unclear. `threadpoolctl` is imported but only installed indirectly through
  scikit-learn. No CI runs the tests (L04 not started).

### R18. Full-cohort runs exhaust memory or time (Operational)

- **Risk:** the full cohort (1.55 million hours) is processed in memory.
- **In place:** Parquet preparation writes in batches; the new shared scorer
  scores 1,000 stays at a time.
- **Remaining exposure:** `train` builds full feature tables in memory and the
  threshold search re-summarizes validation 102 times; not profiled on the
  full cohort.

### R19. Duplicate feature and metric code drifts (Operational)

- **Risk:** features and utility are implemented in `features.py`,
  `missingness.py` and `scripts/causal_features.py`; a later API could add a
  fourth copy, so training and serving disagree.
- **In place:** the architecture review names `features.py` as the single
  contract; the monitoring code reuses `features.py` and `policy.py` rather
  than copying them.
- **Remaining exposure:** not consolidated (J06); no feature-contract version
  in the bundle.

### R20. Monitoring raises false alarms, misses drift, or triggers retraining (Operational)

- **Risk:** about 80 statistics per window cause alarm fatigue for the team;
  real drift is missed; an alarm leads to retraining on test data.
- **In place:** thresholds are the larger of a floor and the no-shift
  99th percentile; fast alarms need two consecutive windows; no alarm on the
  unshifted A test split and correct alarms on simulated shifts (unit change,
  missingness, schema break, new variable, score and calibration shift, lost
  ranking, synthetic hospital B) in the tests; no automatic retraining, written into every reference and summary;
  named owners and a runbook ([monitoring-plan.md](monitoring-plan.md)).
- **Remaining exposure:** the real-data reference is not built; owners and
  floors await team review; the stream is simulated, so gradual drift is not
  exercised.

### R21. Cloud cost and retained resources (Operational)

- **Risk:** a deployment incurs charges; retained S3, KMS, ECR and DynamoDB
  resources stay billable after teardown.
- **In place:** nothing deployed; optional budget alert; documented in the
  [infra README](../infra/cloudformation/README.md).
- **Remaining exposure:** budget alerts do not cap spend. Accepted while
  undeployed.

### R22. Work breaks the course rules (Compliance)

- **Risk:** the course brief does not allow deep learning models. Task H05
  ("Optional: GRU") is a recurrent neural network.
- **In place:** the main models are classical (logistic regression, random
  forest, gradient boosting); sequence models appear only as future work on
  the lessons slide.
- **Remaining exposure:** H05 is still listed. Recommend dropping it, or
  confirming with the instructor before any work.

### R23. Deck or report makes unsupported or inconsistent claims (Communication)

- **Risk:** slides state results that were not run, or contradict each other
  and the repository.
- **In place:** placeholders are marked `[__]`; the design page and deck label
  cloud work as a design.
- **Remaining exposure (found 2026-11-08):** results slides are all
  placeholders; the cover claims warnings "up to 6 hours before sepsis onset";
  the lessons slide mentions an insurance and claims-triage use case
  unrelated to this project; slide owners differ between the issues and
  TEAM-TASKS (Mashie: 10, 12, 15, 16) and the deck README (13, 15, 19, 20); the
  closing slide refers to a Docker demo that does not exist yet (L03). To be
  resolved in M06 and M07.

### R24. Scores read as a diagnosis or a validated six-hour risk (Clinical misuse)

- **Risk:** a reader treats the research score as clinical decision support.
- **In place:** the [clinical-use plan](clinical-use-design-and-validation-plan.md)
  sets release gates; the replay and monitoring outputs carry research-only
  banners; the README and deck state "decision support only; not a medical
  device".
- **Remaining exposure:** the model card (M04) is not written.

### R25. Blocked dependencies delay evaluation (Delivery)

- **Risk:** B replay (M01), explanations (M03), the model card (M04) and the
  results slides all wait for the frozen main model (K03-K05), which has not
  started; submission is due 6 Dec 2026.
- **In place:** monitoring (M02) and this register (M05) were built ahead,
  with synthetic tests, so they only need the frozen bundle to run.
- **Remaining exposure:** schedule; agree a model-freeze date at the next team
  check.

## Change log

| Date | Change | By |
|---|---|---|
| 2026-11-08 | First version: 25 risks from a full repository review | Mashie |
