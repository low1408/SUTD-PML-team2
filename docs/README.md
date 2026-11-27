# Documents: where to look

Start with the first three if you want to run or demonstrate the AWS deployment.

## Run and operate (current design: ECS Fargate, MLflow and Evidently on ECS, Lambda scoring)

| Document | Read it to |
|---|---|
| [RUN-ORDER.md](RUN-ORDER.md) | know what to click, in which order, one page |
| [WORKFLOWS.md](WORKFLOWS.md) | see what every GitHub workflow and option does |
| [cicd.md](cicd.md) | understand what CI runs for a change and how deploy plans its stages |
| [ecs-runbook.md](ecs-runbook.md) | run the ECS demo: power, approval, demo, monitoring; deploy states and fixes |
| [ecs-training.md](ecs-training.md) | check quotas, cost and security notes for training on Fargate |
| [deploy-to-any-account.md](deploy-to-any-account.md) | set up a new AWS account or environment (one-time steps) |
| [serving.md](serving.md), [monitoring-plan.md](monitoring-plan.md), [ward-replay.md](ward-replay.md) | serving, drift monitoring and the ward replay |

## Design, assurance and risk

| Document | Read it to |
|---|---|
| [design/sepsis-warning-pipeline.html](design/sepsis-warning-pipeline.html) | see the whole design as pictures (open in a browser) |
| [architecture-review-and-design.md](architecture-review-and-design.md), [mlops-audit.md](mlops-audit.md) | read the architecture review and the MLOps audit |
| [iam-review.md](iam-review.md), [aws-assumptions.md](aws-assumptions.md) | check permissions, and what was verified against real AWS behaviour |
| [risk-register.md](risk-register.md), [clinical-use-design-and-validation-plan.md](clinical-use-design-and-validation-plan.md) | see the risks and the plan for clinical validation |

## Data and models

[data-profile.md](data-profile.md), [hospital-a-vs-b.md](hospital-a-vs-b.md), [data-versioning.md](data-versioning.md),
[mlflow.md](mlflow.md), [sirs-failure-patterns.md](sirs-failure-patterns.md), [h03-ablation-plan.md](h03-ablation-plan.md).

## Legacy (kept for reference)

[aws-serverless.md](aws-serverless.md): the first design, with SageMaker training and the SageMaker MLflow App. No
environment uses it now.

For the code map see [agent-repository-guide.md](agent-repository-guide.md); the project overview is the top-level
[README](../README.md).
