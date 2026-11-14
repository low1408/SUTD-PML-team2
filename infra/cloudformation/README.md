# AWS research-demo infrastructure

This is a **not-yet-deployed** CloudFormation prototype for the research
scoring service. By default it follows the private design in the deck
(slide 18) and in `docs/design/sepsis-warning-pipeline.html`, but it covers
only part of that design so far:

```text
VPC (or, later, the hospital VPN)
    -> internal Application Load Balancer in private subnets (default)
    -> ECS Fargate scoring task in private subnets
    -> encrypted S3 model artifacts and DynamoDB stay state
```

`LoadBalancerScheme=internet-facing` switches to a public load balancer in
public subnets, for a synthetic-only demo. The API has no authentication in
either mode, so never send patient, personal or other sensitive data to it.
The design targets `ap-southeast-1` (Singapore); the templates use the region
selected for the stack and do not enforce it.

## Deck design vs these templates

| Deck component (slide 18) | In the templates | Status |
|---|---|---|
| Private entry point | Internal ALB in private subnets (`LoadBalancerScheme=internal`, default) | Aligned |
| Site-to-Site VPN from the hospital | Not created; add the VPN range to `AllowedSourceCidr` when one exists | Not built |
| Private API Gateway with OAuth2 client credentials | Not created; the ALB is the entry point and there is no authentication | Not built |
| ECS Fargate scoring API | `service.yaml`, private subnets, image from ECR | Aligned |
| DynamoDB state per stay | `StateTable` in `artifacts.yaml`, task role can read and write it | Aligned (the API does not use it yet) |
| ECR with image scanning | `EcrRepository` with scan on push | Aligned |
| S3 for data and artifacts, KMS encryption | `ArtifactBucket` and `ArtifactKey` | Aligned |
| SageMaker training job and managed MLflow | Not created | Not built |
| EventBridge monitoring jobs | Not created | Not built |
| CloudWatch alarms and SNS email | `monitoring.yaml` | Aligned |
| IAM least privilege | Scoped task and execution roles | Aligned |
| Secrets Manager, CloudTrail | Not created | Not built |

"Not built" items stay on the slide as design, not as deployed resources.

## Layout

| File | Responsibility |
|---|---|
| `root.yaml` | Nested-stack composition and shared parameters |
| `network.yaml` | VPC, public/private subnets, routing, and private AWS service endpoints |
| `artifacts.yaml` | KMS key, private/versioned S3 bucket, scanned ECR repository, encrypted DynamoDB state |
| `service.yaml` | Load balancer (internal by default), private Fargate service, scoped task roles, and application logs |
| `monitoring.yaml` | ECS/ALB alarms, optional email subscription, and optional AWS Budget |

Nested stacks keep resource groups independently reviewable. The templates
create no AWS resources until a stack is explicitly deployed. The root stack
defaults to provisioning only the network and artifact foundation. Enable
`DeployScoringService` only after an image has been built and pushed to the
generated ECR repository; this avoids ECS trying to pull an image that does not
exist yet.

## Local validation

Run the static template linter from the repository root:

```bash
cfn-lint -t infra/cloudformation/*.yaml --ignore-checks W3002
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
