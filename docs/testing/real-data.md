# Real-data testing

This guide exercises the implemented pipeline with the PhysioNet Challenge 2019
training data. It complements the unit tests; it does not replace them.

The pipeline is a research workflow, not a clinically validated system. Keep
downloaded patient records, model bundles, predictions, and reports in the
git-ignored `data/` and `artifacts/` directories. Do not commit or redistribute
those files.

## Environment

Run these commands from the repository root with the Python 3.12 project
environment:

```powershell
$py = ".\.venv312\Scripts\python.exe"
$env:MPLBACKEND = "Agg"
```

If the environment has not been installed:

```powershell
py -3.12 -m venv .venv312
& $py -m pip install -r requirements.txt
& $py -m pip install -e ".[serve,test,checks]"
```

On macOS or Linux, use a POSIX shell and the following equivalent setup:

```bash
python3.12 -m venv .venv
PY=".venv/bin/python"
export MPLBACKEND=Agg
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e ".[serve,test,checks]"
```

If `python3.12` is not available, install Python 3.12 with the operating
system's package manager or select the installed 3.12 interpreter explicitly.

## Staged real-data run

Start with a small, reproducible sample. The download script samples up to 500
records from each hospital with the default seed.

```powershell
& $py scripts\download_data.py `
  --limit 500 `
  --seed 42 `
  --output data\physionet-sample

& $py -m sepsis_warning prepare `
  --data data\physionet-sample `
  --output data\physionet-sample\combined.parquet

& $py -m sepsis_warning split `
  --data data\physionet-sample\combined.parquet `
  --output data\physionet-sample\splits.csv

& $py -m sepsis_warning validate `
  --data data\physionet-sample\combined.parquet `
  --splits data\physionet-sample\splits.csv `
  --output artifacts\data-checks-sample
```

If the sample contains too few septic stays for a stable split, increase
`--limit`. Never change the seed when comparing runs unless the experiment
explicitly calls for a new split.

The macOS/Linux equivalents use forward slashes and `\` for line continuation:

```bash
"$PY" scripts/download_data.py \
  --limit 500 \
  --seed 42 \
  --output data/physionet-sample

"$PY" -m sepsis_warning prepare \
  --data data/physionet-sample \
  --output data/physionet-sample/combined.parquet

"$PY" -m sepsis_warning split \
  --data data/physionet-sample/combined.parquet \
  --output data/physionet-sample/splits.csv

"$PY" -m sepsis_warning validate \
  --data data/physionet-sample/combined.parquet \
  --splits data/physionet-sample/splits.csv \
  --output artifacts/data-checks-sample
```

## Baseline and model evaluation

Run the transparent SIRS comparator before training a model:

```powershell
& $py -m sepsis_warning baseline `
  --data data\physionet-sample\combined.parquet `
  --output artifacts\baseline-sample `
  --figures reports\figures\sample
```

Then train and evaluate the implemented models:

```powershell
& $py -m sepsis_warning train `
  --data data\physionet-sample `
  --output artifacts\real-sample `
  --seed 42 `
  --splits-summary data\physionet-sample\splits.csv.summary.json
```

Additional implemented analyses:

```powershell
& $py -m sepsis_warning sirs-failures `
  --data data\physionet-sample\combined.parquet `
  --output artifacts\real-sample\sirs-failures

& $py -m sepsis_warning compare-hospitals `
  --data data\physionet-sample\combined.parquet `
  --splits data\physionet-sample\splits.csv `
  --output reports\figures\real-sample

& $py -m sepsis_warning missingness `
  --data data\physionet-sample `
  --output artifacts\real-sample\missingness
```

The macOS/Linux equivalents are:

```bash
"$PY" -m sepsis_warning baseline \
  --data data/physionet-sample/combined.parquet \
  --output artifacts/baseline-sample \
  --figures reports/figures/sample

"$PY" -m sepsis_warning train \
  --data data/physionet-sample \
  --output artifacts/real-sample \
  --seed 42 \
  --splits-summary data/physionet-sample/splits.csv.summary.json

"$PY" -m sepsis_warning sirs-failures \
  --data data/physionet-sample/combined.parquet \
  --output artifacts/real-sample/sirs-failures

"$PY" -m sepsis_warning compare-hospitals \
  --data data/physionet-sample/combined.parquet \
  --splits data/physionet-sample/splits.csv \
  --output reports/figures/real-sample

"$PY" -m sepsis_warning missingness \
  --data data/physionet-sample \
  --output artifacts/real-sample/missingness
```

Review these outputs:

- `artifacts\baseline-sample\baseline_metrics.csv`
- `artifacts\real-sample\metrics.json`
- `artifacts\real-sample\validation_models.csv`
- `artifacts\real-sample\held_out_predictions.csv`
- `artifacts\real-sample\replay.html`
- `data\physionet-sample\download_manifest.json`
- `data\physionet-sample\splits.csv.summary.json`

Report validation, internal-test, and external-test results separately. At a
minimum, inspect precision, recall, F1, average precision, AUROC, normalized
utility, false alerts per non-septic patient-day, and early-detection recall.
Do not tune a threshold, model, feature, or preprocessing step using
`external_test`; it represents held-out Hospital B data.

## Full-cohort run

Only run the full cohort after the staged run succeeds and the storage and
runtime have been approved:

```powershell
& $py scripts\download_data.py `
  --limit 0 `
  --seed 42 `
  --output data\physionet-full
```

On macOS/Linux:

```bash
"$PY" scripts/download_data.py \
  --limit 0 \
  --seed 42 \
  --output data/physionet-full
```

Repeat the `prepare`, `split`, `validate`, `baseline`, and `train` commands
using `data\physionet-full` and new output directories such as
`artifacts\real-full`. Do not overwrite an existing dataset or experiment
directory; the CLI intentionally refuses non-empty output destinations.

The full cohort is approximately 40,336 stays and 1.55 million ICU-hours.
Feature construction is memory-intensive, so monitor available RAM and disk
space. Preserve the download manifest, split summary, package versions, seed,
and output hashes with the private experiment record.

## Interpretation boundary

A successful run demonstrates that the software can process the real cohort and
produce reproducible research outputs. It does not establish clinical accuracy,
generalisation to a new hospital, diagnostic value, or treatment benefit.
