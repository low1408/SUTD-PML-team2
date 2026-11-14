## Standard docs

- Purpose, architecture, implemented capabilities, and research limitations: [README.md](README.md).
- Code map, module ownership, data contracts, CLI commands, and test guidance: [docs/agent-repository-guide.md](docs/agent-repository-guide.md).
- Real-data testing workflow and staged PhysioNet evaluation: [docs/testing/real-data.md](docs/testing/real-data.md).
- Installation, data download/preparation, notebook setup, experiments, and replay: [README.md](README.md).
- Task ownership and slide assignments: [TEAM-TASKS.md](TEAM-TASKS.md). No `CONTRIBUTING.md` or documented branching/commit conventions exist; confirm workflow with the team.

## Non-standard tooling

- This is a Python `src/`-layout package named `sepsis_warning`. Use the project's `.venv/bin/python`, not a different environment with the same `.venv` display name. The README recommends Python 3.12 for the notebook; package metadata alone does not certify newer runtimes.
- Use `uv pip install --python .venv/bin/python -e .` for editable installation when uv is available. Dependencies are declared in [pyproject.toml](pyproject.toml); no dependency lockfile exists.
- Downloaded data belongs under project-root `data/physionet/`, not a hardcoded personal path or an external project's dataset. Run data commands from the project root and use their documented `--data`/`--output` arguments.
- `python -m sepsis_warning prepare` creates combined Parquet, but training and replay still read per-record PSV files. The [notebook](src/notebooks/data_check.ipynb) reads Parquet and locates the checkout before importing the local package.
- The current stack is NumPy, pandas, SciPy, scikit-learn, Joblib, PyArrow, Matplotlib, and unittest. Pandera and DVC are optional installs (E03, E02). MLflow tracking and the FastAPI research service are implemented behind optional extras; LightGBM, Docker packaging, and cloud deployment remain planned or design-only.

## Definition of done

- Add or update tests for changed behavior. Run relevant tests first rather than repeatedly running the full suite.
- Every new or changed feature must include documented testing steps: add or update focused unit/regression tests, record the command used to run them, and update the relevant testing guide when the workflow or expected outputs changes.
- Run the existing suite from the project root when the change spans shared behavior:
  `MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v`
- For feature changes, verify prefix invariance, future-measurement corruption, record-local imputation, and exclusion of labels/metadata. For policy changes, verify cooldown resets, hourly decision versus notification accounting, utility, and censored lead-time handling.
- Verify data preparation preserves original values, missingness, labels, source provenance, and record/hour counts. Never use a synthetic fixture to claim real-cohort performance.
- Verify notebook execution with the selected kernel, not just terminal Python. Clear saved outputs before sharing or committing.
- GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs the unit tests, the causal-feature script and the synthetic demo on pull requests. There is no lint step. Do not claim checks passed without looking at the run.

## Conventions

- Reuse shared modules under [src/sepsis_warning](src/sepsis_warning) for data, features, models, policies, and reports; notebooks are for inspection, not duplicate production pipelines.
- At hour t, use only observations available through t. Forward-fill within each record only; never backfill from future hours. Fit imputation and scaling on training data only.
- Preserve the existing seed-42 record-level design: source A is approximately 55% training, 15% calibration, 15% validation, and 15% internal test; all B is external test. Record-disjoint does not guarantee person-disjoint evaluation.
- Fit models on training, probability calibration on calibration, and select models/thresholds on validation. Freeze decisions before internal-test or external-test evaluation.
- Keep `patient_id`, `hospital`, labels, administrative unit codes, and whole-stay summaries out of model inputs. Do not shift `SepsisLabel` again: it is already shifted six hours early.
- Distinguish `hourly_positive` from cooldown-controlled `alert_event`. Published normalized utility uses hourly decisions; notification burden is a separate metric.
- Persist reproducibility evidence such as seeds, package versions, raw hashes, feature settings, and split manifests in private experiment outputs. Treat first-row-positive stays as left-censored for inferred lead time.

## Guardrails / Do Not

- Do not commit secrets, credentials, tokens, or API keys.
- Do not modify generated or vendored files by hand.
- Do not change CI/CD configuration without owner review. If approved GitHub Actions set up Python or another language.
- Treat raw patient files as read-only. Never commit or upload patient data, model files, patient-level predictions, replay reports, or notebook trajectory outputs. Keep data under ignored `data/` and experiment outputs under ignored `artifacts/`; Git ignores do not prevent cloud synchronization.
- Do not overwrite existing datasets, Parquet files, experiment outputs, or split evidence. Choose a new output destination rather than deleting existing work.
- Get approval before adding dependencies, changing storage/data contracts, split strategies, calibration/alert-policy design, or introducing deployment infrastructure. Task assignments do not approve unspecified implementation choices.
- Do not launch full-cohort downloads, training, heavy inference, or held-out evaluation without explicit scope and resource approval. Prefer small synthetic fixtures and focused tests for routine checks.
- Do not use held-out outcomes to select preprocessing, features, models, thresholds, or drift policies. Report cross-source degradation rather than retuning on B and calling it untouched.
- Do not silently suppress convergence warnings, schema errors, or invalid metrics. Surface failures and investigate them.
- Do not load untrusted Joblib/pickle models; deserialization can execute code.
- Do not claim clinically validated risk, real-time readiness, or diagnostic/treatment benefit. Scores target the supplied benchmark label; inferred onset and hourly availability have retrospective limitations. Resolve the documented license-notice discrepancy before dataset redistribution.

## Ownership / Contact

- **Task owners:** Edwin, Jawad, Kaiyuan, Josh, leo, and Mashie; responsibilities are recorded in [TEAM-TASKS.md](TEAM-TASKS.md).
- **Repository maintainer:** 
- **Contact:** 
