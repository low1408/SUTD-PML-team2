# Workflows: what each option does

Twelve workflows, one page. **Automatic** ones start by themselves; every other workflow is started by hand
(**Actions → the workflow → Run workflow**), and the automatic ones also have a Run workflow button. For the click order see
[RUN-ORDER.md](RUN-ORDER.md); for what CI runs see [cicd.md](cicd.md).

Every manual workflow takes **environment** (`dev`, `staging`, `prod`; staging and prod wait for their reviewers) and **region**
(`environment default`, `ap-southeast-1`, `ap-southeast-2`). They are left out of the tables below.

## 1. The workflows at a glance

| Workflow | Starts | Needs AWS | Changes AWS? | Use it to |
|---|---|---|---|---|
| **tests** | automatically on pull requests and pushes, weekly; also by hand | no | no | Lint, type-check, test, scan; only what the change needs |
| **images** | automatically on a push to main that touches `Dockerfile`, `requirements/`, `src/`, `pyproject.toml`; also by hand | writes images | images only | Build the images ahead of a deploy |
| **deploy** | by hand | yes | **yes** | Create or update the stack, serve a model, start training |
| **pipeline** | by hand | yes | starts, stops a training run | Train on real data; check or stop a run |
| **promote** | by hand | yes | model alias and tags | Record the review of a model version |
| **approve** | by hand | yes | releases the pipeline | Approve or reject the waiting model |
| **power** | by hand | yes | starts and stops services | See, start or stop what bills; open browser access |
| **demo** | by hand | yes | writes a page to S3 | Replay stays through the live API |
| **monitor** | by hand | yes | writes reports to S3 | Build the Evidently reference, make the drift report |
| **verify** | by hand | read-only | no | Check every deployed component |
| **collect-results** | by hand, once, at the very end | read-only | no (opens a pull request) | Pull the aggregate results from S3 into the repo for the report |
| **teardown** | by hand | yes | **deletes** | Plan or perform a removal |

## 2. deploy: option → what happens

| `mode` | `retrain` | `dry_run` | Result |
|---|---|---|---|
| `auto` | `auto` | off | The plan compares this commit with what each stage last did and runs only what is needed: **stack** (images, stack update, wait until MLflow and Evidently are healthy), **retrain** (start a real-data run), **serve** (publish the approved model, update Lambda, smoke test). Then verify and the state record |
| `auto` | `no` | off | Same, but **never** starts training. Safe default after a model exists |
| `auto` | `yes` | off | Same, plus a new training run (skipped if one is already running or the stack does not exist) |
| any | any | **on** | Prints the plan table in the run summary and executes nothing |
| `stack and images only` | n/a | off | Images and stack only. No serving, no training. The first click on a new account |
| `full deploy` | `auto` | off | Stack, then serve the newest approved model. **Fails in seconds** if no approved model is published. Trains only if no model exists |

| Other input | Effect |
|---|---|
| `force_prepare` | With a retrain: prepare the data again even though the stored data has the same settings (after changing the preparation code) |
| `model_version` | With serve: serve this published version. Empty = the newest approved one |
| `model_uri` | SageMaker MLflow App only; ignored when MLflow runs on ECS |

Output: the **Deploy plan** table (stage, yes/no, why) and a **Next** list in the run summary; on failure a diagnosis of stack events and
stopped tasks. A run that finishes green records its state in S3.

## 3. pipeline: option → what happens

| `action` | Result |
|---|---|
| `start` | Builds the images for this commit (reused if they exist), copies the data (PhysioNet only, once), starts the run, **and ends**. The run continues in AWS for hours and emails you at the approval step. The green tick only means "started" |
| `status` | Prints the latest run: its step, age, failure reason, and the Fargate tasks running now (what bills) |
| `stop` | Stops the running run; Step Functions also stops the Fargate task it started |

| Option (with `start`) | Effect |
|---|---|
| `data_source = physionet` | The real data (40,336 stays), copied once from the open-data bucket; the real-data gates apply |
| `data_source = synthetic` | A small generated fixture, a rehearsal. A model from it cannot be promoted with `require_real_data` |
| `data_source = physionet-web` | The real data downloaded over HTTPS instead (slower) |
| `prepare` ticked / unticked | Build the data first / reuse what is in S3. The build skips itself when the stored data has the same source, limit and seed |
| `limit` | PhysioNet only: stays per hospital to use, `0` = all |
| `trials` | Tuning trials per model, `0` = keep the default settings |
| `instance_type` | SageMaker only; ignored on ECS |

## 4. Reviewing and serving

| Workflow | Option | Effect |
|---|---|---|
| **promote** | `model_version` | The version number (digits only). Records reviewer and approver, moves `@production`. Refused if it is not a number, was never registered, or the same principal registered and approves it |
| | `require_real_data` ticked | Refuses a version trained on the synthetic fixture |
| **approve** | `decision = approve` | Releases the waiting run: the one frozen evaluation (hospital A test and hospital B, once), then the publish to `models/vN/` with `registry.json` |
| | `decision = reject` | Ends the run and sends a failure notice; `reason` is stored |
| | `execution` | The run id from the email. Empty = the only run waiting; several waiting gives an error |
| **demo** | `mode = ward` (default) | The replay page: a few stays in beds, scored hour by hour through the live API, written to the private bucket `reports/demo/<time>/`. **At most 24 beds** (a limit in `ward.py` for a readable page) and always the same seeded mix. Each stay is one request, so one prediction-log row |
| | `mode = traffic` | Sends many different stays through `/score` and makes **no page**: only the service's own prediction log grows. Use it to give **monitor** the 200 or more requests it needs |
| | `split = validation` | Hospital A validation stays. Safe before the freeze |
| | `split = internal_test` or `external_test` | Hospital A test or hospital B. **Refused** unless the served model has a recorded frozen evaluation |
| | `beds` | ward only: how many stays are on the ward (1 to 24) |
| | `stays` | traffic only: how many different stays to send (default 300) |
| | `seed` | Which random stays. Another number picks others; the same number repeats the same ones |

## 5. Tools, monitoring, checking

| Workflow | Option | Effect |
|---|---|---|
| **power** | `action = status` | Lists what is on, the browser URLs, the estimated hourly cost |
| | `action = start` | Starts the chosen tools; MLflow also starts its database |
| | `component` | `all`, `mlflow`, `evidently` or `database` |
| | `allow_cidr` | Start only: your address as `x.x.x.x/32` opens the tool pages to it. Empty keeps them closed |
| | `allow_anywhere` | Needed to allow `0.0.0.0/0`. The tools have no login: short demos only |
| | `action = stop` | Closes browser access, stops the services and the database. Only storage is left |
| | `force` | Stop only: also while a pipeline step runs (otherwise refused) |
| **monitor** | `action = reference` | Once per model: builds the Evidently reference (hospital A validation) for the serving bundle |
| | `action = report` | Compares the last `days` of the prediction log with the reference. Writes `report.html` and `metrics.json` to `reports/monitoring/<time>/`, updates the Evidently UI, publishes CloudWatch metrics. Below 200 rows it publishes no drift metrics |
| **verify** | `expect_model` off | Checks every component; no scoring function is a warning |
| | `expect_model` ticked | Also requires the scoring function and a signed `/health` answer. Run it after serving a model |

Output of verify: a PASS, WARN or FAIL line per component in the run summary; the run fails on any FAIL.

## 6. collect-results

Run it **once, after everything else is finished** (training, approval, deploy, demo, monitoring). It reads the private bucket,
copies only aggregate files into the repository and opens a pull request to `main`. Nothing on AWS changes.

| Option | Effect |
|---|---|
| `execution` | The pipeline run id. Empty = the latest run that `SUCCEEDED` |
| `model_version` | The version whose approval record is taken. Empty = the newest published |
| `include_monitoring` ticked | Also takes the newest drift report's `metrics.json` (run **monitor** first) |
| `include_failure_examples` | Also takes `fig_failure_examples.png`, which shows individual stays. Off by default |
| `dry_run` | Lists what would be collected and stops: no branch, no pull request |

| Collected into | What |
|---|---|
| `reports/results/` | The report's tables and figures, `run/` (selection files), `tuning/`, `registry.json` (who approved, the frozen evaluation time), `monitoring/metrics.json`, and `PROVENANCE.md` (run, model, commit, SHA-256 of every file, what was skipped) |
| `reports/results_test/` | `evaluation_metrics.json`, the one frozen test evaluation (aggregates only) |

Never copied: the model, the prepared data, per-patient files (`held_out_predictions.csv`, `split_manifest.csv`), ward replays, the Evidently
`report.html`. The bucket name and the AWS account id are replaced by placeholders in text files. The workflow refuses to commit anything
outside the two folders or any file that looks like patient data or a model.

One-time setting: Settings → Actions → General → **Allow GitHub Actions to create and approve pull requests**. A pull request opened by a
workflow does not start the tests: close and reopen it once. The report generator's test curves and confidence intervals need per-stay
predictions that the pipeline does not keep, so they are not part of this collection.

## 7. teardown

| `scope` | Result |
|---|---|
| `plan only` | Lists what would be removed. Nothing changes |
| `stack only` | Deletes the CloudFormation stack. Models, data, keys, images and logs are kept |
| `everything` | Also deletes both S3 buckets with every version, the image repositories, the table and the log groups; the KMS key is scheduled for deletion in 7 days |

Anything but `plan only` needs `confirm` set to the stack name (`sepsis-dev`, `sepsis-staging` or `sepsis-prod`). `everything` cannot be undone.

## 8. The automatic ones

| Workflow | Trigger | Effect |
|---|---|---|
| **tests** | pull request, push to main | Classifies the change and runs only the needed jobs. `ci-ok` is the one required check |
| | weekly | Known-vulnerability scan of the four lock files |
| | by hand, `run_everything` ticked | Runs every job; unticked compares with main |
| **images** | push to main that touches image inputs | Builds and pushes the images for the commit to ECR. Skips quietly if there is no stack. Repository variable `AUTO_PREBUILD` = `false` turns it off |
