# H03: with/without ablation plan for the HMM and signature features

Owner: Josh. Status: **script ready (PR #71), full-cohort run pending.** Run it with
`scripts/h03_ablation.py`. Results go on slide 14 (H06).

## Question

Do the temporal features add anything on top of the shared feature builder?

- HMM state probabilities (H01, merged in PR #60): `hmm_state_0..2`, forward-filtered.
- Depth-2 log-signature over a 6-hour window (H02, PR #63): 15 `sig_*` columns.

## Arms

| Arm | Features | Question it answers |
|---|---|---|
| A | Shared features only (`features_for_windows`, J06) | Reference |
| B | A + 3 HMM columns | Does the HMM help? |
| C | A + 15 signature columns | Does the signature help? |
| D | A + HMM + signature | Do they add to each other? |

New columns are appended after the shared columns, so A's columns and order are
identical in every arm.

## Held fixed in every arm

| Item | Setting |
|---|---|
| Split | Edwin's `splits.csv` (seed 42). Fit on `train`, report on `validation` |
| Base features | J06 `features_for_windows`, J02 training-only fallback |
| HMM | Fitted once on `train` by `scripts/fit_hmm.py`: 3 states, HR, MAP, Resp, O2Sat. Frozen; only the forward filter is used for features |
| Signature | `fit_signature_scaler` on `train` only; window 6 hours; depth 2 |
| Model | K01 NumPy logistic regression, same regularisation, class weighting and seed |
| Tuning budget | The same hyper-parameter grid for every arm, chosen the same way. No arm gets extra tuning |
| Calibration and threshold | The team pipeline (K04, K05) run identically per arm: calibration fitted on `calibration` with the model frozen; threshold chosen on `validation` under the 0.2 false alerts per nonseptic patient-day budget, 6-hour cooldown |
| Pre-declared choices | 3 HMM states and a 6-hour window are fixed now and are not tuned on validation |

## Metrics (validation, hospital A)

From the team's `summarize()` so every arm is scored the same way:

- Primary: hourly average precision (AUPRC).
- Also: AUROC, Brier score, normalized hourly utility.
- At the chosen threshold: early-detection recall, false alerts per nonseptic
  patient-day, median lead hours.
- **First 6 ICU hours** (ICULOS <= 6), reported separately. SIRS catches only
  39.5% of septic hours there (E06), so this is where history-based features
  could matter most, although there is also the least history to use.

## Uncertainty

Paired bootstrap over validation **stays** (1,000 resamples, the same resamples
for every arm). Report each arm's difference from A with a 95% interval. We say
a feature set "helps" only if the interval for its AUPRC difference is above 0.

## Cost

| Item | How it is measured |
|---|---|
| Extra columns | 3 (HMM), 15 (signature), 18 (both) |
| HMM fit | Measured: 52 s, 25 iterations, 11,183 train stays (435,449 hours) on Josh's laptop |
| Feature time | Seconds per 1,000 stays for each arm, measured on the real data |
| Model fit time | Per arm, same machine |

## Out of scope for H03

- `internal_test` and `external_test` (hospital B) are not used. Any test
  comparison waits until features, model, calibration and threshold are frozen.
- No change to the number of HMM states or the window length after seeing
  validation results.
- GRU (H05) is not run: the course does not allow deep learning.

## What goes on slide 14

One table (4 arms by AUPRC, AUROC, Brier, utility, early recall, first-6-hour
AUPRC) with differences from A and 95% intervals, one cost row, and the
limitations. If the temporal features do not help, the slide says so.

## Dependencies

| Needed | Owner | Status on 14 Nov |
|---|---|---|
| NumPy logistic regression (K01) | Kaiyuan | Merged (PR #68, module `logistic.py`) |
| Calibration and threshold (K04, K05) | Kaiyuan | Platt calibration merged (PR #68); threshold uses `tune_threshold` |
| `summarize()` metrics without scikit-learn, if the library ban covers metrics | To agree | `policy.py` still imports scikit-learn metrics |
| HMM features (H01) | Josh | Merged (PR #60) |
| Signature features (H02) | Josh | Merged (PR #63) |
| Memory for the full run | Josh | `NumpyLogisticRegression` peaks around 6 GB on the full train split; needs a machine with more than 8 GB RAM, or a float32 change to `logistic.py` |

## Open decision: J02 fallback

Applying the J02 training-median fallback to raw values before `features_for_windows`
makes `__missing_now`, `__never_seen` and `__hours_since_seen` constant. The script
leaves it off by default (the model's training-only median imputer fills gaps) and
`--fallback` turns it on. Either way it is the same in every arm. To agree with the team.
