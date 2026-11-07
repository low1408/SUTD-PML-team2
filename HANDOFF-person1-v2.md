# Person 1 (Edwin) — full handoff v2

Everything needed to build Person 1's part of the sepsis early-warning project.
Read top to bottom once, then work task by task (section 9).

**Confidence labels used below**
- ✅ **Verified** — checked in this session (code run, official source read, or
  package tested).
- 🟡 **From memory** — well-known facts, but re-check against the paper before
  putting numbers in the report.

---

## 0. TL;DR

- Data: PhysioNet Challenge 2019, hourly ICU records, hospital **A** (train) and
  **B** (external test). Label `SepsisLabel` is **already shifted 6 h early**.
- Course rules: **no deep learning; models hand-coded in NumPy.** Rules like
  SIRS, data tools (pandas, pyarrow, Pandera, DVC) are fine.
- My job: **SIRS baseline → A-vs-B exploration → Pandera → DVC → tidy-up →
  slides 4, 5, 8.** Target: code done ~20 Nov; final deadline **6 Dec 11:59 pm**.
- Leo already built `prepare` (→ Parquet), `missingness`, and an EDA notebook.
  Reuse them; tell Leo I'm taking SIRS, Pandera, DVC, A-vs-B.

---

## 1. Course requirements (prof's brief + class)

| Rule | Meaning |
|---|---|
| **No deep learning** | No neural nets; no PyTorch / TensorFlow / Keras. |
| **Models hand-coded** | Training loop, loss, predictions in NumPy. |

- ✅ OK: NumPy, pandas, pyarrow, matplotlib/seaborn, scikit-learn for
  **cleaning / splitting / scaling**, Pandera, DVC, FastAPI/Flask.
- ❌ Not as the model: sklearn models, XGBoost, **LightGBM**, CatBoost,
  statsmodels models, AutoML, any neural net (incl. **GRU**).
- Grey area (ask prof): sklearn **metric** functions; library models as a
  labelled comparison only. Safe: hand-write metrics (section 6).

**Deliverables:** code + PDF report on eDimension + **public GitHub link** by
**6 Dec 11:59 pm**; do **not** upload data (link to PhysioNet). Presentation
week 13 (slides + small demo/video). Report 10–15 pages + appendix. Team list
with who did what.

**Marks:** Concept 5% · **Coding 25%** · **Performance & evaluation 20%**
(tuning, **comparison with SIRS**, right metrics) · **Report 30%** ·
Presentation 20% · Bonus +5%.

**Report sections:** 1 exec summary (≤1 page) · 2 background · 3 related work ·
4 problem formulation · **5 data description & exploration (mine)** · 6 methods
+ how to run · 7 evaluation + bias audit · 8 discussion · 9 recommendations ·
10 limitations/future work · 11 optional research ideas. Also: pipeline
diagram, "tried but failed" section, **command to recreate every figure**.

---

## 2. The dataset

Source: https://physionet.org/content/challenge-2019/1.0.0/ (CC BY 4.0 — cite it).
Files: one `.psv` (pipe-separated) per ICU stay, one row per hour.

### 2.1 Size
| | Hospital A (`training_setA`) | Hospital B (`training_setB`) |
|---|---|---|
| Real hospital 🟡 | Beth Israel Deaconess, Boston | Emory University, Atlanta |
| Stays | 20,336 🟡 | 20,000 🟡 |
| Septic stays 🟡 | ~1,790 (~8.8%) | ~1,142 (~5.7%) |

Total ✅ (Leo's run on the real data): **40,336 stays, 1,552,210 hourly rows**.
A-train split ✅: 11,183 stays / 435,449 hours, **2.16% positive hours**.
A hidden third hospital was used for final challenge scoring and is not public.

Note: B has a lower septic share than A (🟡) → **prevalence shift**, one of the
things the A-vs-B exploration should show.

### 2.2 Columns (40 inputs + label) — units per the challenge page 🟡
**Vitals (8):** HR (beats/min), O2Sat (%), Temp (°C), SBP, MAP, DBP (mmHg),
Resp (breaths/min), EtCO2 (mmHg).

**Labs (26):** BaseExcess (mmol/L), HCO3 (mmol/L), FiO2 (fraction — check
whether 0–1 or %), pH, PaCO2 (mmHg), SaO2 (%), AST (IU/L), BUN (mg/dL),
Alkalinephos (IU/L), Calcium (mg/dL), Chloride (mmol/L), Creatinine (mg/dL),
Bilirubin_direct (mg/dL), Glucose (mg/dL), Lactate (listed mg/dL; values look
like mmol/L — check), Magnesium (mmol/dL), Phosphate (mg/dL), Potassium
(mmol/L), Bilirubin_total (mg/dL), TroponinI (ng/mL), Hct (%), Hgb (g/dL),
PTT (s), WBC (×10³/µL), Fibrinogen (mg/dL), Platelets (×10³/µL).

**Demographics (6):** Age (years; very old ages may be capped — check max),
Gender (0 = female, 1 = male), Unit1 (MICU flag), Unit2 (SICU flag),
HospAdmTime (hours between hospital and ICU admission, usually negative),
ICULOS (ICU length of stay in hours, the hour index).

**Label:** SepsisLabel (0/1).

In this repo ✅ (`src/sepsis_warning/data.py`): `VITALS`, `LABS`, `STATIC =
[Age, Gender, HospAdmTime, ICULOS]`. **Unit1/Unit2 deliberately excluded** —
admin codes can act as a hospital shortcut.

### 2.3 Missingness (real data, A-train) ✅
HR ~7.8% missing · Temp ~66% · Lactate ~96.5% · **EtCO2 100%** (useless at A).
Labs are measured every few hours, so most lab cells are NaN. Missing ≠ zero ≠
normal. Whether something is measured reflects care practice (a doctor ordered
lactate because they were worried) — this can differ between hospitals.

---

## 3. How sepsis is labelled (Sepsis-3)

Reference: Reyna et al., *Critical Care Medicine* 2020 (the challenge paper);
Singer et al., *JAMA* 2016 (Sepsis-3).

1. **Suspicion of infection** 🟡 — antibiotic given + blood culture ordered
   (antibiotic first → culture within 24 h; culture first → antibiotic within
   72 h). Earlier one = **t_suspicion**.
2. **Organ damage** 🟡 — **SOFA** score (lungs PaO2/FiO2, clotting platelets,
   liver bilirubin, circulation MAP/vasopressors, brain GCS, kidneys
   creatinine/urine output) rises **≥ 2** from 48 h before to 24 h after
   t_suspicion → **t_SOFA**.
3. **t_sepsis = earlier of t_suspicion and t_SOFA.**
4. ✅ **`SepsisLabel = 1` from t_sepsis − 6 h onward**; never-septic stays are
   0 throughout. The official scorer recovers onset as
   `t_sepsis = first positive row + 6` (code: `np.argmax(labels) - dt_optimal`,
   `dt_optimal = -6`).

**The model never sees** antibiotics, cultures, urine output, GCS or
vasopressors — removed so the answer isn't given away.

**Limitations to state in the report**
- Label comes from treatment timing, not a doctor's diagnosis → partly encodes
  each hospital's habits (how fast they order cultures/antibiotics).
- Stays positive from row 1 began sepsis before ICU → onset unknown; repo
  excludes them from lead-time analysis ✅ (`policy.summarize`,
  `left_censored`).
- Post-onset hours stay labelled 1 although the warning is no longer "early".

---

## 4. Official scoring: the utility function ✅

Verified against the official scorer
(github.com/physionetchallenges/evaluation-2019, `evaluate_sepsis_score.py`):

```
dt_early = -12, dt_optimal = -6, dt_late = 3
max_u_tp = 1, min_u_fn = -2, u_fp = -0.05, u_tn = 0
```

Per hour t, relative to onset t_sepsis (septic patients):
| Hours from onset | Predict 1 | Predict 0 |
|---|---|---|
| ≤ −12 (too early) | −0.05 | 0 |
| −12 … −6 | rises 0 → +1 | 0 |
| −6 … +3 | falls +1 → 0 | falls 0 → −2 |
| > +3 | 0 (ignored) | 0 (ignored) |

Non-septic patients: every predicted 1 = −0.05, predicted 0 = 0.

**Normalised utility** = (observed − inaction) / (best − inaction), summed over
all patients; inaction = always predict 0 (scores 0 by definition); best =
predict 1 exactly from onset−12 to onset+3.

✅ The repo's `policy.utility_components` / `normalized_utility` matches these
equations. **Utility uses hourly predictions**, not the cooldown-suppressed
alert events.

Challenge results 🟡: 104 teams, 853 entries ✅ (CinC abstract). Winning
entry used **signature features** + gradient boosting (Morrill et al., "Can I
get your signature?"); best full-test utility ≈ 0.36 🟡, much lower on the
unseen third hospital — **generalising to a new hospital is hard**, which is
exactly our project question. Cite this in related work (verify numbers).

---

## 5. SIRS baseline (Task 1) — exact rule

SIRS (Sepsis-1, Bone et al. 1992 / ACCP-SCCM consensus). ≥ 2 of 4:

| Criterion | Rule with dataset columns |
|---|---|
| Temperature | `Temp > 38` **or** `Temp < 36` (°C) |
| Heart rate | `HR > 90` |
| Breathing | `Resp > 20` **or** `PaCO2 < 32` |
| White cells | `WBC > 12` **or** `WBC < 4` (×10³/µL); bands >10% not in data |

Rules for our implementation:
- **Causal:** use the **last observed value so far** (forward-fill within the
  patient). Never backfill, never use later hours.
- **Missing (never observed yet) = not abnormal** (comparisons with NaN are
  False — make this explicit in a test).
- Output per hour: `sirs_count` (0–4) and `sirs_alert = count >= 2`.
- Score SIRS as a binary predictor (hourly_positive = sirs_alert). For AUROC/AP,
  use `sirs_count / 4` as a score.
- Run it through the same `policy.apply_policy` (6 h cooldown) and
  `policy.summarize` as the models, plus precision/recall/F1.

Verified snippet ✅ (run on synthetic data; future-hour corruption leaves
earlier alerts unchanged):

```python
def sirs(frame):
    last = frame[["Temp", "HR", "Resp", "PaCO2", "WBC"]].ffill()
    count = (((last.Temp > 38) | (last.Temp < 36)).astype(int)
             + (last.HR > 90).astype(int)
             + ((last.Resp > 20) | (last.PaCO2 < 32)).astype(int)
             + ((last.WBC > 12) | (last.WBC < 4)).astype(int))
    return count, (count >= 2).astype(int)
```

**Caveat:** forward-filling a WBC from 20 hours ago keeps it "abnormal" for a
long time. Optional variant: only carry labs forward for N hours (e.g. 24) —
report it as a sensitivity check.

**What to expect** 🟡 (literature): SIRS is **sensitive but not specific**
(meta-analysis, mortality outcome: sensitivity ~0.82, specificity ~0.24) →
many false alarms, likely low precision and negative/near-zero utility in ICU
where most patients meet SIRS often. That's the story: *does our model beat
the rule that fires on almost everyone?*

### Optional extra baselines (bonus marks, still rules → allowed)
**qSOFA (partial):** Resp ≥ 22, SBP ≤ 100, altered mentation (**GCS not in
data → only 2 of 3 criteria**). Alert if both available criteria met. Report
as "partial qSOFA".

**NEWS2 (partial)** 🟡 — points per parameter, alert at total ≥ 5 (≥ 7 urgent):
| Parameter | 3 | 2 | 1 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| Resp | ≤8 | | 9–11 | 12–20 | | 21–24 | ≥25 |
| SpO2 (O2Sat, scale 1) | ≤91 | 92–93 | 94–95 | ≥96 | | | |
| SBP | ≤90 | 91–100 | 101–110 | 111–219 | | | ≥220 |
| HR | ≤40 | | 41–50 | 51–90 | 91–110 | 111–130 | ≥131 |
| Temp | ≤35.0 | | 35.1–36.0 | 36.1–38.0 | 38.1–39.0 | ≥39.1 | |
Supplemental O2 = +2 (proxy: FiO2 > 0.21 if available); consciousness not in
data.

Literature context 🟡: qSOFA = specific but insensitive; NEWS in between;
SIRS = sensitive but unspecific.

---

## 6. Metrics to report (hand-coded, NumPy)

Hourly (labels y ∈ {0,1}, predictions p ∈ {0,1}):
```
TP = sum(p & y); FP = sum(p & ~y); FN = sum(~p & y); TN = rest
precision = TP / (TP + FP)      recall = TP / (TP + FN)
F1 = 2·precision·recall / (precision + recall)
```
- **AUROC** (from a score s): probability a random positive hour scores above a
  random negative hour = Mann–Whitney: rank all scores (average ranks for
  ties), `AUROC = (sum ranks of positives − n₊(n₊+1)/2) / (n₊·n₋)`.
- **Average precision:** sort by score desc, `AP = Σ (recall_k − recall_{k−1})
  · precision_k`.
- Already in repo ✅: normalised utility, false alerts per non-septic
  patient-day, early-detection recall (alert within 12 h before onset), median
  lead time, Brier score.
- Patient-level view too: "was this septic patient alerted before onset?"
- Report on **validation, internal test (A), external test (B)** separately.

Imbalance: ~2% positive hours → always-predict-0 gets ~98% accuracy and F1 = 0.
**Never headline accuracy.**

---

## 7. Pandera schemas (Task 3)

Package ✅: `pandera==0.35.0` (latest at time of writing; requires Python ≥3.10).
New import style: `import pandera.pandas as pa`.

Principle: reject **physically impossible / malformed** data; **flag, don't
drop** unusual-but-possible values (matches Leo's notes).

Verified sketch ✅ (passes on synthetic data; catches HR=999 and Gender=3 with
`lazy=True` reporting all failures at once):

```python
import pandera.pandas as pa
from pandera.typing import Series

class PatientHours(pa.DataFrameModel):
    HR: Series[float] = pa.Field(ge=0, le=300, nullable=True)
    O2Sat: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    Temp: Series[float] = pa.Field(ge=20, le=45, nullable=True)
    Resp: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    WBC: Series[float] = pa.Field(ge=0, nullable=True)
    Age: Series[float] = pa.Field(ge=0, le=120)
    Gender: Series[float] = pa.Field(isin=[0, 1])
    ICULOS: Series[float] = pa.Field(ge=1)
    SepsisLabel: Series[float] = pa.Field(isin=[0, 1])

    class Config:
        strict = False   # allow the other columns; add them all in the real file
        coerce = True

    @pa.dataframe_check
    def label_never_drops(cls, df):
        return bool((df["SepsisLabel"].diff().dropna() >= 0).all())

    @pa.dataframe_check
    def hours_consecutive(cls, df):
        return bool((df["ICULOS"].diff().dropna() == 1).all())

PatientHours.validate(frame, lazy=True)   # raises pa.errors.SchemaErrors
```

To do in the real `schemas.py`:
- Add **all 40 columns** with ranges. Suggested impossible-value bounds (tune
  after looking at real min/max in Leo's notebook): SBP/MAP/DBP 0–300,
  EtCO2 0–100, pH 6.5–8.0, FiO2 0–1 (or 0–100 — check), SaO2 0–100,
  Lactate ≥0, Platelets ≥0, all labs ≥0 except BaseExcess (can be negative),
  HospAdmTime ≤ 0 mostly (check — may have positives), Unit1/Unit2 ∈ {0,1}.
- Per-patient checks (label never drops, consecutive hours) run per file; on
  the combined Parquet, run them grouped by `patient_id`.
- A second "soft" check that only **reports** outliers (e.g. HR > 250, Temp <
  30) into a CSV for the EDA, without failing.
- Call it inside `prepare` before writing Parquet; existing hand-written checks
  in `data.read_patient` can stay or call the schema.
- Tests: valid synthetic passes; each kind of broken data raises with a clear
  message naming the column.
- Add `"pandera>=0.35,<1"` to `pyproject.toml` dependencies.

---

## 8. DVC (Task 4)

Package ✅: `dvc==3.67.1`. Install **inside the venv**: `pip install dvc`
(verified installs cleanly in a fresh venv; if it fails on Python 3.14 on
Windows, create the venv with Python 3.12: `py -3.12 -m venv .venv`).

What DVC does here: records **fingerprints (hashes)** of the data and each
step's outputs in `dvc.lock` (committed to Git), so anyone can rebuild the
exact same data with `dvc repro`. The data itself stays out of Git.

Verified `dvc.yaml` ✅ (DVC parses it and builds the chain download → prepare →
split):

```yaml
stages:
  download:
    cmd: python scripts/download_data.py --limit 0 --output data/physionet
    deps:
      - scripts/download_data.py
    outs:
      - data/physionet/training_setA
      - data/physionet/training_setB
      - data/physionet/download_manifest.json
  prepare:
    cmd: python -m sepsis_warning prepare --data data/physionet --output data/processed/combined.parquet
    deps:
      - data/physionet/training_setA
      - data/physionet/training_setB
      - src/sepsis_warning/prepare.py
      - src/sepsis_warning/schemas.py
    outs:
      - data/processed/combined.parquet
      - data/processed/combined.parquet.summary.json
  split:
    cmd: python -m sepsis_warning split --data data/processed/combined.parquet --output data/processed/splits.csv --seed 42
    deps:
      - data/processed/combined.parquet
      - src/sepsis_warning/data.py
    outs:
      - data/processed/splits.csv
```

Steps:
1. `pip install dvc` → `dvc init` → commit `.dvc/`, `.dvcignore`.
2. Write `dvc.yaml` (above). **The `split` command does not exist yet** — build
   it: read Parquet, one row per patient (`patient_id`, `hospital`,
   `septic = max(SepsisLabel)`), reuse `split_patients` logic, write
   `splits.csv` (patient_id, hospital, split). Training then loads this file
   instead of re-splitting; Leo's notebook can drop its fake-B-patient hack.
3. `prepare` currently refuses to overwrite its output. DVC deletes outputs
   before rerunning a stage, so this is fine — but check it.
4. `dvc repro` → commit `dvc.yaml` + `dvc.lock`.
5. Optional remote so teammates `dvc pull` instead of re-downloading: a shared
   folder or Google Drive (`pip install dvc-gdrive`; Drive auth can be fiddly —
   if it fights you, skip; `dvc repro` re-downloading is acceptable).
6. README section: `pip install -e . dvc` → `dvc repro` → done.

Notes: `/data/` is already in `.gitignore` ✅. `dvc repro --dry` shows what
would run. Changing `schemas.py` re-runs `prepare` automatically.

---

## 9. Task list with "done looks like"

| # | Task | Done looks like |
|---|---|---|
| 1 | **SIRS baseline** — `baselines.py`, CLI `baseline`, tests | One command → `baseline_metrics.csv` with SIRS + always-no rows for validation / internal test / B: precision, recall, F1, AUROC, AP, utility, false alerts per patient-day, early-detection recall. Tests: exact firing hour; future-hour invariance; missing ≠ abnormal. README section. |
| 2 | **A-vs-B exploration** | Logic in `src/`, notebook only imports. Figures in `reports/figures/`, each with its recreate command. ~5–6 figures + summary table + 3–4 headline sentences. |
| 3 | **Pandera** | `schemas.py` with all 40 columns; runs in `prepare`; soft-outlier report; tests with broken data; pandera in deps. |
| 4 | **DVC** | `dvc.yaml` + `dvc.lock` committed; new `split` command writes `splits.csv`; `dvc repro` works on a fresh clone; README section. |
| 5 | **Tidy-up** | `train`/`baseline` read Parquet + `splits.csv`; `prepare` gives a clean error (CLI catches only `ValueError`, `prepare` raises `FileExistsError` ✅ verified bug); README data path; `.venv-*/` added to `.gitignore`; update `contributions.md`. |
| 6 | **Slides 4, 5, 8** | Built from real numbers (section 11). |

### A-vs-B figures (Task 2)
1. Stays, hours, septic share per hospital (bar + table).
2. Stay length distribution A vs B (histogram, log x).
3. Age distribution + gender share A vs B (needed for the bias audit later;
   suggested age groups 18–44 / 45–64 / 65–79 / 80+).
4. **Missingness per variable A vs B** (paired horizontal bars) — likely the
   biggest difference; supports "care practices differ".
5. Vital-sign distributions A vs B (HR, MAP, Temp, Resp; density overlay).
6. SIRS firing rate A vs B (share of hours with SIRS ≥ 2; septic vs non-septic).
7. Optional: hours from ICU admission to first positive label, A vs B.

Rules: describe B **descriptively only** — don't tune anything on B. Compute
everything from the Parquet with grouped operations (fast). Clear
patient-level notebook outputs before committing.

---

## 10. Repo facts to know ✅

- Package `src/sepsis_warning/` — **keep the name**.
- `data.split_patients(patients, seed)`: A split stratified by ever-septic,
  ~55/15/15/15 (train/calibration/validation/internal_test); B = external_test;
  checks no overlap; needs ≥10 septic and ≥10 non-septic in A.
- `policy.apply_policy(table, threshold, cooldown)` needs columns
  `patient_id`, `hour`, `probability` → adds `hourly_positive`, `alert_event`.
  For SIRS pass `probability = sirs_alert` and threshold 0.5.
- `policy.summarize(table)` needs `label`, `probability`, `hourly_positive`,
  `alert_event`.
- `features.META = ["patient_id", "hospital", "hour", "label"]`.
- Default false-alert budget 0.2 per non-septic patient-day; cooldown 6 h.
- Leo: `prepare.py` (validates + writes Parquet with `patient_id`, `hospital`
  columns, zstd), `missingness.py`, `src/notebooks/data_check.ipynb`,
  `docs/1.understand-training-data.md`.
- `modeling.py` uses sklearn models → not allowed; Person 3 (Kaiyuan) replaces.
- Tests: `python -m unittest discover -s tests` (12 pass on my laptop).
- Commands: `demo` (synthetic), `train`, `replay`, `missingness`, `prepare`.

---

## 11. Slides 4, 5, 8 (draft content — confirm outline with Mashie)

- **Slide 4 — Data & task:** PhysioNet 2019, 40,336 ICU stays, 1.55 M hours,
  40 inputs; Sepsis-3 label shifted 6 h early; ~2% positive hours (imbalance).
  One small timeline graphic of the label.
- **Slide 5 — A vs B:** 2–3 strongest differences (septic share, missingness,
  age/length). Message: B is a genuinely different hospital → good transfer test.
- **Slide 8 — Split & baseline:** patient-level split diagram (A 55/15/15/15,
  B untouched); SIRS vs always-no table (precision, recall, F1, utility, false
  alerts/day). Message: "SIRS fires on almost everyone — this is the bar."

## 12. Report section 5 (my part) — outline
5.1 Source & licence · 5.2 Cohorts (table A vs B) · 5.3 Variables & units ·
5.4 Label definition & 6 h shift (+ limitations) · 5.5 Missingness ·
5.6 Data quality checks (Pandera rules, outliers flagged) · 5.7 Split strategy
& leakage protection · 5.8 Reproducibility (DVC, hashes, seeds).
Also contribute: SIRS baseline method (section 6) and its results (section 7);
"tried but failed" entries from my tasks.

## 13. Working notes (Windows)
- Activate first: `.venv\Scripts\Activate.ps1` (made with `py -m venv .venv`;
  Python 3.14.2). If you see `.venv-1`, delete it (VS Code created it).
- Branch `edwin/person1-data`; `git pull origin main` regularly; commit per
  task; PR into `main` at the end with a teammate review.
- Build on synthetic data first (`python -m sepsis_warning demo`), then real.
  Real download: `python scripts/download_data.py --limit 0 --output
  data/physionet` (try `--limit 500` first if slow).
- Pin exact versions at the end for the brief: `pip freeze > requirements.txt`.
- Explain each piece as it's built — I must present it.

## 14. References
- Reyna MA et al. Early Prediction of Sepsis From Clinical Data: the
  PhysioNet/Computing in Cardiology Challenge 2019. *Crit Care Med* 2020.
  https://www.cinc.org/2019/Program/accepted/412.html
- PhysioNet challenge-2019 dataset: https://physionet.org/content/challenge-2019/1.0.0/
- Official scorer: https://github.com/physionetchallenges/evaluation-2019
- Singer M et al. The Third International Consensus Definitions for Sepsis and
  Septic Shock (Sepsis-3). *JAMA* 2016.
- Bone RC et al. Definitions for sepsis and organ failure (SIRS). *Chest* 1992.
- Meta-analysis qSOFA vs SIRS vs NEWS: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9012380/
- SIRS vs qSOFA vs NEWS in the ED: https://ajemjournal.com/article/S0735-67571830889-1/fulltext
- Signature method winning entry (Morrill et al.): https://www.maths.ox.ac.uk/node/35939
- Pandera docs: https://pandera.readthedocs.io/ · lazy validation:
  https://pandera.readthedocs.io/en/stable/lazy_validation.html
- DVC pipelines: https://doc.dvc.org/user-guide/pipelines/running-pipelines
