# AWS research-demo infrastructure

This is a **not-yet-deployed** CloudFormation prototype for the research
platform. The default is serverless and pay-per-use:

```text
SageMaker AI training job (spot, on demand) -> serverless SageMaker MLflow App (@candidate, @production)
    -> published bundle in encrypted S3 -> Lambda scoring function behind an IAM-authenticated function URL
```

`ServingMode=fargate` selects the deck's always-on private design instead
(slide 18 and `docs/design/sepsis-warning-pipeline.html`):

```text
VPC (or, later, the hospital VPN)
    -> internal Application Load Balancer in private subnets (default)
    -> ECS Fargate scoring task in private subnets
    -> encrypted S3 model artifacts and DynamoDB stay state
```

In Fargate mode `LoadBalancerScheme=internet-facing` switches to a public load
balancer for a synthetic-only demo, and the API has no authentication. In
Lambda mode every request must be SigV4-signed. Never send patient, personal or
other sensitive data to either until access and data handling are approved.
Both `ap-southeast-1` (Singapore) and `ap-southeast-2` (Sydney) are supported and
linted in CI; the templates use the region selected for the stack and do not enforce it.

## Deck design vs these templates

| Deck component (slide 18) | In the templates | Status |
|---|---|---|
| Private entry point | Lambda mode: function URL with IAM (SigV4) authentication, not network-private. Fargate mode: internal ALB in private subnets | Partly (choose by `ServingMode`) |
| Site-to-Site VPN from the hospital | Not created; add the VPN range to `AllowedSourceCidr` when one exists | Not built |
| Private API Gateway with OAuth2 client credentials | Not created. Lambda mode authenticates callers with IAM instead; Fargate mode has no authentication | Not built |
| ECS Fargate scoring API | `serverless.yaml` Lambda by default; `service.yaml` Fargate with `ServingMode=fargate` | Aligned (Lambda default) |
| DynamoDB state per stay | `StateTable` in `artifacts.yaml`, task role can read and write it | Aligned (the API does not use it yet) |
| ECR with image scanning | `EcrRepository` with scan on push | Aligned |
| S3 for data and artifacts, KMS encryption | `ArtifactBucket` and `ArtifactKey` | Aligned |
| SageMaker training job and managed MLflow | `mlops.yaml`: serverless MLflow App and training role; jobs started by `scripts/sagemaker_train.py` (spot) | Aligned (serverless) |
| EventBridge monitoring jobs | Not created | Not built |
| CloudWatch alarms and SNS email | `monitoring.yaml` | Aligned |
| IAM least privilege | Scoped task and execution roles | Aligned |
| Secrets Manager, CloudTrail | Not created | Not built |

"Not built" items stay on the slide as design, not as deployed resources.

## Layout

| File | Responsibility |
|---|---|
| `root.yaml` | Nested-stack composition, `ServingMode` (lambda by default, or fargate) and shared parameters |
| `artifacts.yaml` | KMS key, private/versioned S3 bucket with lifecycle rules, three scanned ECR repositories (scoring-api, scoring-lambda, training) that keep 10 images each, encrypted on-demand DynamoDB state |
| `mlops.yaml` | Serverless SageMaker AI **MLflow App**, the SageMaker training-job role, and a team operator policy |
| `serverless.yaml` | **Lambda** scoring function (container image), IAM-authenticated function URL, logs (default serving mode) |
| `network.yaml` | VPC, subnets, routing and VPC endpoints (Fargate mode only) |
| `service.yaml` | Internal load balancer, private Fargate service, task roles, logs (Fargate mode only) |
| `monitoring.yaml` | Lambda or ECS/ALB alarms, optional email subscription, optional AWS Budget (always created) |
| `pipeline.yaml` | Step Functions training pipeline: train with gates → human approval → frozen evaluation → publish |
| `observability.yaml` | Prediction log to S3 (log subscription + Firehose), daily drift Lambda, drift alarm, optional retraining trigger |
| `apigw.yaml` | Optional private API Gateway with IAM auth, throttling and WAF in front of the Lambda |

The default stack is serverless and pay-per-use: nothing bills by the hour.
Training jobs are started on demand by `scripts/sagemaker_train.py`, and
`scripts/aws_deploy.sh` runs the whole workflow (see
[docs/aws-serverless.md](../../docs/aws-serverless.md), which also has a cost
estimate). Enable `DeployScoringService` only after an image has been pushed and
a model published to S3 (`ModelS3Uri`, `ModelSha256`).

## Local validation

Run the static template linter from the repository root:

```bash
pip install "cfn-lint[sam]"
cfn-lint -t infra/cloudformation/*.yaml --regions ap-southeast-1 ap-southeast-2 --ignore-checks W3002
```

`W3002` is suppressed because the root template intentionally uses local
nested-template paths, which the AWS `package` command resolves later. This
checks CloudFormation structure and resource properties without AWS
credentials or provisioning. It does not prove that the stack deploys in a
particular account, that an image is runnable, or that the API behaves
correctly. Those require a later, approved validation phase.

Do not run `aws cloudformation package` or deploy commands as part of this
prototype: packaging uploads nested templates to S3, and deployment creates
billable resources. No AWS upload, stack creation, or teardown has been done.

## Before any future deployment

- Review the design and obtain explicit owner approval for the account, region,
  budget, retention, and teardown plan.
- First provision the foundation stack with the default
  `DeployScoringService=false`, then build and test the FastAPI image, push it
  to the generated ECR repository, and update the stack with
  `DeployScoringService=true` and its immutable image tag as `ScoringImageUri`.
  The image must expose port 8000 and a non-sensitive `GET /health` response.
- Configure an ACM certificate for a custom hostname and set the matching
  `ApiHostname` parameter to enable HTTPS. Point that hostname at the load
  balancer. Without a certificate/hostname the prototype serves HTTP; it is
  for synthetic demonstration traffic only. The image is built and
  smoke-tested by `scripts/docker_smoke_test.sh` (see `docs/serving.md`).
- Keep `LoadBalancerScheme=internal` unless a public synthetic demo is agreed.
  `AllowedSourceCidr` defaults to the VPC range; add only the hospital VPN range.
  There is no request authentication or rate limiting.
- Set a nonzero `MonthlyBudgetUsd` and `BudgetEmail` together to create a budget
  alert. AWS Budgets notifications are not a hard spending cap.
- Do not upload raw PhysioNet records, patient-level outputs, secrets, or
  registry artifacts. The bucket is a future artifact location, not permission
  to move project data to AWS. Confirm license, privacy, residency, and owner
  approval before any data upload.
- Review application logging: request bodies and patient-level values must not
  be written to CloudWatch Logs.

## Cost, monitoring, and recovery

The ALB, Fargate tasks, interface VPC endpoints, CloudWatch Logs/alarms, KMS,
and any stored S3/ECR/DynamoDB data can incur charges. There is no reliable
monthly estimate until the account, region, runtime, traffic, and retention
are agreed. The optional budget sends threshold notifications but cannot stop
spend.

CloudWatch alarms cover task CPU/memory, target errors/latency, and healthy
targets. They do not monitor model drift or clinical performance. ECS deployment
circuit-breaker rollback is enabled; CloudFormation rolls back failed stack
updates. The S3 bucket, KMS key, ECR repository, and DynamoDB table use retain
policies, so teardown intentionally requires a separate data-retention review
and may leave billable resources behind.
