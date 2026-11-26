# IAM review: does every role have what its code needs?

Method: each role's policy in `infra/cloudformation/` was read statement by statement and compared with what the code that runs as that
role calls (S3 prefixes, KMS, DynamoDB, CloudWatch, ECS, SNS). `tests/test_iam_grants.py` now keeps the result honest: it spells out what
each role needs and fails when a template drops it, and ties two values to constants in the code (the metrics namespace, the prefix the
Evidently UI lists).

## Role by role

| Role (template) | Runs | Needs | Verdict |
|---|---|---|---|
| Training task (`ecs-platform`) | prepare, tune, train, evaluate and publish, monitor reference | read `data/ tuning/ raw/ models/`; write `data/ tuning/ reports/ training-output/ models/ monitoring/evidently/`; list; KMS | **Gap found and fixed** (#46): could not write `monitoring/evidently/`, so monitor reference failed |
| Approver task (`ecs-platform`) | promote | write `training-output/`; KMS; MLflow over HTTP | OK, and deliberately cannot publish, train or write data |
| ECS execution (`ecs-platform`) | pull the training image, write logs | ECR pull on the training repository, log group | OK |
| MLflow server (`mlflow-ecs`) | artifact proxy | `mlflow-artifacts/` read, write, delete; list; KMS | OK. Execution role reads only its own database secret |
| Evidently report (`evidently-ecs`) | monitor report | read `prediction-log/ monitoring/evidently/`; write `reports/monitoring/` and the workspace; metrics in `SepsisWarning/Evidently` | OK on paper; not yet run, because the reference step failed first |
| Evidently UI (`evidently-ecs`) | live pages | read and list the workspace prefix; KMS | OK. Read only |
| Lambda scoring (`serverless`) | the API | read `models/`; DynamoDB get and update; KMS; logs | OK. Cannot write to S3 |
| Firehose and log subscription (`observability`) | prediction log delivery | write `prediction-log/`; KMS | OK on paper; delivery not yet checked in a run |
| Drift function (`observability`) | daily drift job | read the log and `monitoring/`; write `drift-results/`; logs | OK |
| Step Functions pipeline (`pipeline-ecs`) | the training run | RunTask, RegisterTaskDefinition, Stop and Describe, events rule, PassRole for the task and execution roles, SNS publish | OK, proven by a full run. PassRole does not include the approver role |
| GitHub deploy role (`github-oidc`) | every workflow | `AdministratorAccess` by default (`DeployPolicyArns`) | Works everywhere. Broad; see below |

## Findings

1. **Fixed in #46:** the training task role could not save the Evidently reference.
2. **Fixed here:** the drift alarm's retrain rule (`RetrainOnDriftRule`) started the pipeline with `image_tag`, `git_commit` and
   `require_real_data` only, but the ECS pipeline reads `prepare`, `data_source`, `limit`, `trials` and `force` too, so the first step
   would have failed. The rule is off unless `RetrainImageTag` is set; its input now carries every key. A test lists them.
3. **Not a gap, but broad:** the deploy role is `AdministratorAccess`. That is why every workflow worked on the first try. For a shared
   or production account replace it through `DeployPolicyArns` with a policy scoped to CloudFormation, ECS, ECR, S3, IAM pass-role and
   the services in the stack.
4. **`Resource: "*"` is where AWS requires it:** `ecr:GetAuthorizationToken`, `ecs:RegisterTaskDefinition`, and
   `cloudwatch:PutMetricData` (limited by a namespace condition). Everything else names a prefix or an ARN.
5. **Separation of duties holds in the policies:** the approver role cannot write models, data or reports, and the pipeline role cannot
   pass the approver role.

## Not exercised in a real run yet

The Evidently report and reference path, delivery of the prediction log by Firehose, the CloudWatch alarms publishing to the alert
topic, the drift function, and calls to the function URL by a principal other than the deploy role (`AllowedInvokerArn`). The first run
of each should be read from its log.

The people-facing managed policies in `mlops.yaml` (trainer, approver, deployer) date from the SageMaker design. The workflows do not
use them: they sign in as the deploy role.
