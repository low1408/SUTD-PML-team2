# Handoff: Person 1 (Edwin) — sepsis early-warning project

Context carried over from a planning session. Read this first, then start on
**Task 1 (SIRS baseline)** below.

---

## 1. The project in one paragraph

Hourly early-warning model for sepsis on the **PhysioNet Challenge 2019** data
(https://physionet.org/content/challenge-2019/1.0.0/). One `.psv` file per ICU
stay, hourly vitals + labs + demographics, label `SepsisLabel`. Question: **does a
model trained on hospital A (training_setA) keep useful performance on hospital B
(training_setB) under a limited alert budget?** A is split by patient into
~55% train / 15% calibration / 15% validation / 15% internal test. All of B is an
untouched external test set.

Key data facts (from `docs/1.understand-training-data.md`, real data):
- 40,336 stays, 1,552,210 hourly rows; 40 inputs + label.
- `SepsisLabel` is **already shifted 6 h early** and stays 1 afterwards. Never
  shift it again. Inferred onset = first positive hour + 6.
- Only ~2.2% of training hours are positive → accuracy is useless; use
  precision / recall / F1 / AP.
- Heavy missingness: HR ~8%, Temp ~66%, Lactate ~97%, EtCO2 100% missing.
- Features at hour t may only use hours ≤ t of the same patient (causal).

### How sepsis is labelled (Sepsis-3, Reyna et al. 2020)
The `MIT-LCP/physionet` repo is only a download tool; the labelling comes from
the challenge paper. Organisers applied the **Sepsis-3** definition
(**infection + organ damage**) to full hospital records:
1. **Suspicion of infection** — antibiotic given + blood culture ordered close
   together (antibiotic first → culture within 24 h; culture first → antibiotic
   within 72 h). Earlier of the two = **t_suspicion**.
2. **Organ damage** — **SOFA** score (lungs PaO2/FiO2, clotting platelets,
   liver bilirubin, circulation MAP/vasopressors, brain GCS, kidneys
   creatinine/urine) rises **≥ 2 points** between 48 h before and 24 h after
   t_suspicion. That time = **t_SOFA**.
3. **Onset t_sepsis = earlier of t_suspicion and t_SOFA.**
4. **`SepsisLabel = 1` from t_sepsis − 6 h onwards**; 0 before; 0 always for
   patients who never meet the definition.

Hospitals: **A = Beth Israel Deaconess (Boston), B = Emory (Atlanta)**; a third
hidden hospital is not public.

The model **does not see** antibiotics, cultures, urine output, GCS or
vasopressors (removed to avoid giving the answer away) — it must pick up
physiological warning signs before clinicians act.

**Why SIRS is the baseline:** SIRS is the older (1990s "Sepsis-1") bedside rule
— easy to compute, many false alarms. Slide 8 story: *does our model beat the
rule doctors already use?*

**Limitations for the report:**
- The label is computed from treatment timing, not a doctor's diagnosis, so it
  partly reflects how fast each hospital orders tests/antibiotics — this may
  differ between A and B (link to the A-vs-B exploration).
- Stays positive from their first row started sepsis before the ICU stay; onset
  unknown → excluded from lead-time analysis.
- ~7% of stays ever septic, ~2% of hours positive.

**Cite:** Reyna et al., "Early Prediction of Sepsis From Clinical Data: The
PhysioNet/Computing in Cardiology Challenge 2019", *Critical Care Medicine*,
2020; Singer et al., Sepsis-3, *JAMA*, 2016.

## 2. Course rules (from the prof's brief + class)

| Rule | Meaning |
|---|---|
| **No deep learning** | No neural nets at all; no PyTorch / TensorFlow / Keras. |
| **Models hand-coded** | Training loop, loss and predictions written in NumPy. |

- ✅ Allowed: NumPy, pandas, pyarrow, matplotlib/seaborn, scikit-learn for
  **cleaning, splitting, scaling**, Pandera, DVC, FastAPI/Flask.
- ❌ Not allowed as the model: scikit-learn models, XGBoost, LightGBM, CatBoost,
  statsmodels models, AutoML.
- Grey area (ask prof): sklearn **metric** functions; library models as a
  labelled comparison only. Safe choice: hand-write metrics (few lines).
- The **SIRS baseline is a rule, not a trained model**, so it's fine.

### Deadlines and deliverables
- **6 Dec, 11:59 pm** — code + PDF report on eDimension + **public GitHub link**.
  Do **not** upload the dataset; link to PhysioNet instead.
- **Week 13** — presentation (slides + small demo/video).
- Report 10–15 pages (excl. appendix/refs) + team list with who did what.

### Marks
Concept 5% · **Coding 25%** · **Performance & evaluation 20%** (tuning,
comparison with **SIRS baseline**, right metrics) · **Report 30%** ·
Presentation 20% · Bonus +5%.

Report sections: exec summary, background, related work, problem formulation,
**data description & exploration**, methods, evaluation + bias audit,
discussion, recommendations, limitations/future work. Also: pipeline diagram,
"tried but failed" section, and **for every figure, the command that recreates it**.

## 3. Team (from `contributions.md`, on branch `feature/data-cleanup-documentation`)

| Person | Owner | Tasks | Slides |
|---|---|---|---|
| 1 | **Edwin (me)** | Download→Parquet, DVC, Pandera, A/B split, EDA, SIRS baseline | 4, 5, 8 |
| 2 | Jawad | Causal rolling features, missing values, hours since last lab, leakage tests, utility score, shared feature module | 6 |
| 3 | Kaiyuan | Logistic regression, random forest, ~~LightGBM~~, calibration, threshold + 6 h cooldown, fairness | 3, 7, 9 |
| 4 | Josh | HMM forward states, signature features, with/without tests, ~~GRU~~ | 11 |
| 5 | Leo | MLflow, FastAPI, Docker, CI, latency, AWS design | 13, 14 |
| 6 | Mashie | Hospital B replay, drift monitoring, SHAP, model card, risk register, deck | 10, 12, 15, 16 |

Rule conflicts the team must fix: LightGBM and GRU are not allowed; the current
sklearn models in `modeling.py` and the sklearn `LogisticRegression` in
`missingness.py` must become hand-coded NumPy (Person 3's job, not mine).

**Leo already did part of my role** (Parquet + EDA). Tell Leo I'm taking SIRS,
Pandera, DVC and the A-vs-B comparison.

## 4. Current repo state (as of `main` @ 6fe083d)

Package: `src/sepsis_warning/` (keep this name — renaming breaks teammates).

| File | What it does |
|---|---|
| `data.py` | Reads `.psv`, basic checks (columns, binary nondecreasing labels, consecutive `ICULOS`), `split_patients` (patient-level, stratified by septic, A 55/15/15/15, B external), synthetic data generator |
| `features.py` | Causal features, single 6 h window (last value, missing now, never seen, hours since seen, rolling mean/std/count, change) |
| `modeling.py` | ⚠ sklearn models + `CalibratedClassifierCV` — not allowed, Person 3 replacing |
| `policy.py` | Threshold, 6 h cooldown alerts, official utility score, metrics (`summarize`), threshold tuning under a false-alert budget (0.2 per non-septic patient-day) |
| `cli.py` | Commands: `demo`, `train`, `replay`, `missingness`, `prepare` |
| `prepare.py` (Leo) | Validates all `.psv` and writes one `combined.parquet` (+ summary JSON) |
| `missingness.py` (Leo) | Missingness report + compares fill strategies (uses sklearn LR ⚠) |
| `src/notebooks/data_check.ipynb` (Leo) | EDA on A-train only: labels, stay lengths, missingness, HR/MAP/Temp/Resp histograms, one example trajectory |
| `scripts/download_data.py` | Downloads from PhysioNet (`--limit 0` = all) |
| `scripts/causal_features.py` (Jawad) | Standalone draft of features; not wired into the package yet |
| `docs/1.understand-training-data.md` (Leo) | Key data facts |

Useful commands:
```
python scripts/download_data.py --limit 0 --output data/physionet
python -m sepsis_warning prepare --data data/physionet --output data/physionet/combined.parquet
python -m sepsis_warning missingness --data data/physionet --output artifacts/missingness
python -m sepsis_warning demo            # synthetic smoke test
python -m unittest discover -s tests     # 12 tests, all pass
```
Alternative downloader: `pip install physionet` then
`physionet download challenge-2019 --version 1.0.0 --include "training/*" --output data/physionet`.

`data/` and `artifacts/` are git-ignored — never commit patient data.

## 5. My tasks — status and plan

| ID | Task | Status |
|---|---|---|
| E01 | Download + Parquet | Mostly done (Leo). Gaps in Task 5. |
| E02 | DVC | Not started |
| E03 | Pandera schemas | Not started (hand-written checks exist in `data.py`) |
| E04 | A/B split | Done (`split_patients`) |
| E05 | Exploratory analysis | Partly done (Leo). Missing A-vs-B comparison. |
| E06 | **SIRS baseline** | Not started — most important |
| E07 | Slides 4, 5, 8 | Not started |

### Task 1 — SIRS baseline (do first)
- New `src/sepsis_warning/baselines.py`.
- **SIRS rule**, causal (forward-filled last observed value, never future hours).
  Count criteria true at each hour:
  - Temp > 38 or < 36 °C
  - HR > 90
  - Resp > 20 **or** PaCO2 < 32
  - WBC > 12 **or** < 4 (×10⁹/L)  — band count not in dataset
  - **≥ 2 criteria → alert.** Missing (never observed) = not abnormal.
- **Always-no** baseline.
- CLI `python -m sepsis_warning baseline --data ...`: score validation,
  internal test and hospital B with **precision, recall, F1**, utility,
  false alerts per non-septic patient-day, early-detection recall — same 6 h
  cooldown via `policy.apply_policy` / `summarize`. Write `baseline_metrics.csv`.
- Tests: hand-made patient fires at the exact expected hour; changing a future
  hour never changes earlier alerts; missing values never count as abnormal.
- Done = one command → table with SIRS + always-no rows per split; tests pass;
  README section. Models later add rows to the same table.

### Task 2 — Hospital A vs B exploration
- Logic in `src/`, notebook only imports it (e.g. `notebooks/hospital_comparison.ipynb`).
- Compare A vs B: septic share, stay length, age & gender (needed for bias
  audit), missingness per variable, vital-sign distributions, SIRS firing rate.
- Save figures to `reports/figures/` with the recreate command for each.
- Done = ~5–6 figures + 1 summary table + 3–4 headline sentences (slide content).

### Task 3 — Pandera schemas
- `src/sepsis_warning/schemas.py`: columns/types, **physically impossible**
  ranges only (e.g. HR 0–300, Temp 20–45, Gender 0/1), labels 0/1
  nondecreasing, consecutive hours per patient.
- Run inside `prepare` so bad data stops before Parquet is written.
- Add `pandera` to `pyproject.toml`. Tests with deliberately broken data.
- Flag unusual-but-possible values; never auto-clip/remove (matches Leo's notes).

### Task 4 — DVC
- `dvc.yaml` stages: `download → prepare → split`; commit `dvc.lock`.
- Save the split once as `splits.csv` (patient_id, hospital, split) so everyone
  uses the same split and Leo's notebook can drop its fake-B-patient workaround.
- README: `dvc repro` on a fresh clone reproduces everything; data stays off GitHub.

### Task 5 — Tidy-up
- `train` / `baseline` read `combined.parquet` instead of the `.psv` files.
- `prepare` raises `FileExistsError` when output exists → CLI only catches
  `ValueError`, so it crashes with a traceback. Show a clean message instead.
- README: install → download → prepare → validate → split.
- Update my rows in `contributions.md`.

### Task 6 — Slides 4, 5, 8 (after real numbers exist)
Likely: **4** dataset + label + imbalance · **5** A-vs-B differences +
missingness · **8** patient-level split + SIRS results table models must beat.
Check the deck outline with Mashie.

## 6. Working notes
- Windows: `python` only works inside the venv. Activate first:
  `.venv\Scripts\Activate.ps1` (create with `py -m venv .venv`). Python 3.14.2.
- Work on branch `edwin/person1-data`; `git pull` from `main` regularly; commit
  after each task; open a PR into `main` at the end and get a teammate review.
- Build/test on synthetic data first (`python -m sepsis_warning demo`), then run
  on real data. If the full download is slow, try `--limit 500` first.
- Explain each piece as it's built — I need to present it (concept marks).
- Target: code done ~20 Nov so others can use the baseline and split before 6 Dec.
