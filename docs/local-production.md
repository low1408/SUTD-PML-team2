# Production-like local stack

`infra/local/compose.yaml` runs the scoring service in the same layout as the AWS
stack in `infra/cloudformation/`: a load balancer in front of two
Fargate-sized tasks, a private network with no internet, a read-only
container, and the model taken from the registry's `@production` alias with
its hash checked. `scripts/local_prod.sh` starts it and runs the checks. Use
synthetic or local models only.

## What maps to what

| AWS (CloudFormation) | Local stand-in | Same? |
|---|---|---|
| ECR image built from `Dockerfile` | `sepsis-warning:local`, same Dockerfile, built for x86_64 | Same image |
| ECS Fargate task: 0.5 vCPU, 1 GB, X86_64, non-root | `api-a`, `api-b`: `cpus: 0.5`, `mem_limit: 1g`, `linux/amd64`, uid 10001 | Same limits. Fargate CPU is not the same speed as a laptop |
| Two private subnets, no public IP, no NAT | `private` Docker network with `internal: true`; tasks publish no port | Same effect: no inbound except through the load balancer, no outbound |
| Internal ALB, HTTP 80, target group checks `GET /health` (200-399) | HAProxy on `127.0.0.1:8080` with active `/health` checks, round robin | Same behaviour; check interval shortened from 30 s to 2 s |
| Rolling deploy, `MinimumHealthyPercent: 100`, ALB deregistration, circuit breaker | `local_prod.sh deploy`: drains one task at the load balancer, replaces it, waits until it is healthy, then does the other | Same order; shorter drain (2 s, AWS default 300 s); no automatic rollback |
| S3 `models/` prefix (KMS), task role `s3:GetObject` | Bundle copied from the MLflow registry to `artifacts/local-prod/vN/`, mounted read-only | Different source; see the gaps below |
| CloudWatch Logs | `bash scripts/local_prod.sh logs` (size-capped) | Logs only; no alarms |

## Run it

Needs Docker Desktop (or Docker Engine with Compose v2), and the venv from the
README with `pip install -e ".[serve,test,tracking]"`.

```bash
source .venv/bin/activate

# 1. A model in the registry (synthetic demo), reviewed and promoted
python -m sepsis_warning demo --output artifacts/demo \
  --mlflow-uri sqlite:///artifacts/mlflow.db --register-model
python -m sepsis_warning promote --version 1 --reviewed-by "Leo"

# 2. Stage @production, build the image, start the load balancer and two tasks
bash scripts/local_prod.sh up
curl -s http://127.0.0.1:8080/health

# 3. Production checks (about 30 s)
bash scripts/local_prod.sh test
```

`test` checks, all through the load balancer:

- `/health` reports the staged bundle's hash, and `/score` returns valid probabilities;
- `/model` shows the selected model and its validation-only comparison, and `/predict` scores a ward batch;
- requests are spread across both tasks;
- input that contains the label gets a 422;
- tasks have no published port and cannot reach the internet;
- tasks run as non-root, with a read-only root filesystem and model volume;
- a task with the wrong bundle hash refuses to start;
- with one task stopped, every request still succeeds;
- p50/p95/p99 latency (saved to `artifacts/local-prod/latency.json`).

## Release and roll back

```bash
# Train and register a second version, review it, promote it
python -m sepsis_warning demo --data data/demo2 --output artifacts/demo2 --seed 43 \
  --mlflow-uri sqlite:///artifacts/mlflow.db --register-model
python -m sepsis_warning promote --version 2 --reviewed-by "Leo"

# Rolling update under live traffic; fails if any request fails
bash scripts/local_prod.sh deploy models:/sepsis-warning@production

# Roll back to a version you name
bash scripts/local_prod.sh deploy models:/sepsis-warning/1

bash scripts/local_prod.sh down
```

In testing, five rollouts in a row (versions 2, 1, 2, 1, 2) had 0 failed
requests out of about 100 each. Without the drain step, about 1 request in 50
failed while a task was being replaced.

## Settings

| Variable | Default | Use |
|---|---|---|
| `PLATFORM` | `linux/amd64` | Matches Fargate. On an Apple Silicon Mac this runs under emulation and is slower; use `linux/arm64` for speed, not for latency numbers |
| `ALB_PORT` | `8080` | Local port of the load balancer |
| `MLFLOW_URI` | `sqlite:///artifacts/mlflow.db` | Registry the model is taken from |
| `BASE_IMAGE` | `python:3.12-slim` | Use `mirror.gcr.io/library/python:3.12-slim` if Docker Hub rate-limits |
| `EXTRA_CA_CERT` | none | CA file for a TLS-inspecting proxy, passed as a build secret only |
| `PYTHON` | `python` | Interpreter with the `tracking` extra installed |

The CloudFormation templates themselves can be checked locally with
`cfn-lint -t infra/cloudformation/*.yaml --ignore-checks W3002`; `W3002` only
notes that the nested templates need `aws cloudformation package`.

## What this does not cover

- **Model source.** The task definition gives the container `MODEL_BUCKET`
  and `MODEL_PREFIX`, but the API does not read them yet: it loads a file at
  `/models/model.joblib`. In AWS something still has to put the approved
  bundle there (an init container that runs `aws s3 cp`, or code in the API
  that reads S3) and pass `--model-sha256`. Locally the script does that
  step from the MLflow registry. This gap needs fixing before a real deployment.
- **DynamoDB stay state.** The table exists in CloudFormation but the API is
  stateless (each request carries the full stay), so there is nothing to emulate.
- **TLS, IAM, KMS, security groups.** Not emulated. The load balancer is
  plain HTTP on localhost; the CloudFormation HTTPS listener needs an ACM
  certificate.
- **Alarms, budget, autoscaling.** Not emulated; logs only.
- **Speed.** Laptop CPUs differ from Fargate's, so latency here is a sanity
  check, not a capacity figure. Measure on AWS before agreeing on a target.
