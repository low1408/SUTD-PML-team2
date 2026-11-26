# Run order: what to click, in which order

Everything is started from **GitHub → Actions → the workflow → Run workflow**. Waiting is most of the time (about 3.5 hours for a
first real run). The one rule: **deploy → pipeline → promote → approve → deploy → demo → collect-results → stop**.

## 0. Once per AWS account and region (not in GitHub)

1. From an admin shell for that account: `AWS_REGION=<region> GITHUB_REPO=<owner/name> bash scripts/bootstrap_github_oidc.sh`.
   It creates the GitHub login (no stored keys) and the deploy roles, and sets `AWS_ACCOUNT_ID`, `AWS_REGION` and `AWS_DEPLOY_ROLE_ARN`.
2. In the GitHub environment (`dev`, `staging`, `prod`) set `BUDGET_EMAIL` and `ALERT_EMAIL`. For `staging` and `prod` add required
   reviewers: that is the approval gate.
3. Check `infra/params/<environment>.json`. All three files now run on ECS (no SageMaker quota) with Spot for the long steps.

## 1. The sequence

| # | Workflow | Inputs | What it does | Done when |
|---|---|---|---|---|
| 1 | **deploy** | `mode = auto` | Creates the infrastructure and the images (15 to 25 min) | Verify is green, or only the Lambda line is red on a first run |
| 2 | **pipeline** | `action = start`, `data_source = physionet`, `prepare` ticked, `trials = 6` | Copies the real data once, tunes, trains, writes the report | **pipeline** `status` says `WaitForApproval` (2 to 3 h) |
| 3 | review | MLflow UI, `reports/<run>/` in S3 | Check metrics, real-data tag and the report | You are satisfied |
| 4 | **promote** | `model_version = 1` (digits only) | Records the reviewer and moves `@production` | It succeeds |
| 5 | **approve** | `decision = approve` | Runs the one frozen evaluation and publishes the model | **pipeline** `status` says `SUCCEEDED` |
| 6 | **deploy** | `mode = auto`, `retrain = no` | Serves the model, smoke test, verify, records the state | Green |
| 7 | **demo**, then **monitor** | `mode = ward`, `split = validation`; then `mode = traffic`, `stays = 300`; then **monitor** `reference`, then `report` | The ward page shows alerts; the traffic run gives the 200 or more requests the drift report needs | The Evidently UI shows the project |
| 8 | **collect-results** | `dry_run` first, then unticked | Pulls the aggregate results from S3 and opens a pull request for the report | The pull request exists; merge it after looking at the figures |
| 9 | **power** | `action = stop` | Stops everything that bills | `status` shows only storage |

Between steps: **pipeline** `action = status` shows the step and what is running. **power** `action = status` prints the browser
URLs; they change every time a service restarts.

## 2. Things that trip people up

- **A green pipeline workflow only means the run started.** The run itself takes hours; read **pipeline** `status`.
- **promote before approve.** Approve releases the pipeline, and its next step refuses a model that was not promoted.
- **`model_version` is a plain number** (`1`, not `v1`). It is shown in the approval email and in the MLflow UI.
- **Do not run deploy or power stop while the pipeline is in Prepare, Tune, Train or EvaluateAndPublish.** A stack update restarts
  MLflow, which the run still needs.
- **Use `dry_run` on deploy first** if you are unsure: it prints the plan and runs nothing.
- **`retrain = no`** on the second deploy guarantees no new multi-hour training starts.
- **MLflow or Evidently URL stopped loading:** the address changed. **power** `status` shows the new one; **power** `start` with
  `allow_cidr` (your address as `x.x.x.x/32`) reopens browser access.

## 3. Cost in one table

| What | Costs while | How to keep it low |
|---|---|---|
| Training task (4 vCPU, 16 GB) | the run (about 2 h) | **Tune and Train already run on Fargate Spot** (`EcsUseSpot=true`), AWS advertises up to 70% off. Prepare and the final evaluation are short and stay on demand |
| MLflow, Evidently UI, database | they are on, about $0.10 an hour together | **power** `stop` when you are not using them: this is the biggest saving |
| Lambda scoring | per request | nothing to do |
| Storage, keys, images, logs | always, a few dollars a month | `teardown` removes the stack when the project is over |

A Spot step that AWS reclaims has no automatic retry: start **pipeline** again with `prepare` unticked (the prepared data is reused).
Set `EcsUseSpot` to `false` in the parameter file for on-demand only.

More detail: `docs/ecs-runbook.md` (components, costs, failure scenarios), `docs/cicd.md` (what runs automatically),
`docs/deploy-to-any-account.md` (the one-time setup).
