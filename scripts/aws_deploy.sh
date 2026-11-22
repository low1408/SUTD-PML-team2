#!/usr/bin/env bash
# Serverless AWS workflow for the research platform (see docs/aws-serverless.md).
#
#   bash scripts/aws_deploy.sh stack                 create/update the stack (storage, MLflow App, roles, budget)
#   bash scripts/aws_deploy.sh images                build and push the lambda, training and scoring images
#   bash scripts/aws_deploy.sh data [DIR]            upload combined.parquet + splits.csv (+ summary) for training
#   bash scripts/aws_deploy.sh train [ARGS...]       SageMaker training job on spot; registers @candidate
#   bash scripts/aws_deploy.sh mlflow-ui             print a short-lived link to the MLflow App UI
#   bash scripts/aws_deploy.sh promote VERSION NAME  move @production after review
#   bash scripts/aws_deploy.sh publish               copy @production to s3://<bucket>/models/vN/
#   bash scripts/aws_deploy.sh serve                 deploy the published model on Lambda (or Fargate)
#   bash scripts/aws_deploy.sh call METHOD PATH [JSON]  signed request to the scoring API
#   bash scripts/aws_deploy.sh pipeline              start the Step Functions pipeline (train -> approval -> evaluate -> publish)
#   bash scripts/aws_deploy.sh approve TOKEN         approve a waiting pipeline (the token is in the notification email)
#   bash scripts/aws_deploy.sh reference FILE        upload a monitor-setup reference and switch on the daily drift job
#   bash scripts/aws_deploy.sh outputs | down
#
# ENVIRONMENT=dev|staging|prod picks infra/params/<env>.json and the stack sepsis-<env>
# (use one AWS account per environment). prod needs CONFIRM_PROD=yes.
# promote refuses synthetic models unless ALLOW_SYNTHETIC=1 (dev rehearsals).
#
# Settings (environment variables): STACK (sepsis-research), AWS_REGION
# (ap-southeast-1), SERVING_MODE (lambda | fargate), BUDGET_USD and BUDGET_EMAIL
# (monthly budget alert), ALERT_EMAIL, PLATFORM (linux/amd64), TAG (git short sha),
# PYTHON (python). Needs the AWS CLI v2, Docker, and `pip install -e ".[aws,tracking,sagemaker]"`.
#
# Everything here is billable only while it runs or stores data: no load balancer,
# NAT gateway, VPC endpoint or MLflow server runs between uses in lambda mode.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENVIRONMENT="${ENVIRONMENT:-}"
if [ -n "$ENVIRONMENT" ]; then
  [ -f "$ROOT/infra/params/$ENVIRONMENT.json" ] || { echo "no infra/params/$ENVIRONMENT.json" >&2; exit 2; }
  STACK="${STACK:-sepsis-$ENVIRONMENT}"
fi
STACK="${STACK:-sepsis-research}"
export AWS_REGION="${AWS_REGION:-ap-southeast-1}"
SERVING_MODE="${SERVING_MODE:-lambda}"
PLATFORM="${PLATFORM:-linux/amd64}"
PYTHON="${PYTHON:-python}"
TAG="${TAG:-$(git -C "$ROOT" rev-parse --short HEAD)}"
STATE="$ROOT/artifacts/aws"
mkdir -p "$STATE"
fail() { echo "FAIL: $*" >&2; exit 1; }

output() {  # output NAME -> stack output value (empty if absent)
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue | [0]" --output text 2>/dev/null | sed 's/^None$//'
}

account() { aws sts get-caller-identity --query Account --output text; }

cmd_stack() {
  local extra=("$@")
  local templates_bucket="$STACK-$(account)-$AWS_REGION-cfn"
  if ! aws s3api head-bucket --bucket "$templates_bucket" 2>/dev/null; then
    echo "Creating template bucket $templates_bucket"
    aws s3api create-bucket --bucket "$templates_bucket" \
      --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
    aws s3api put-public-access-block --bucket "$templates_bucket" --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  fi
  aws cloudformation package --template-file "$ROOT/infra/cloudformation/root.yaml" \
    --s3-bucket "$templates_bucket" --s3-prefix "$STACK" --output-template-file "$STATE/packaged.yaml" >/dev/null
  local params=("ServingMode=$SERVING_MODE")
  if [ -n "$ENVIRONMENT" ]; then
    [ "$ENVIRONMENT" != prod ] || [ "${CONFIRM_PROD:-}" = yes ] || fail "prod changes need CONFIRM_PROD=yes"
    while IFS= read -r line; do params+=("$line"); done < <("$PYTHON" -c '
import json, sys
for key, value in json.load(open(sys.argv[1])).items():
    if not key.startswith("_"):
        print(f"{key}={value}")' "$ROOT/infra/params/$ENVIRONMENT.json")
  fi
  [ -n "${BUDGET_USD:-}" ] && params+=("MonthlyBudgetUsd=$BUDGET_USD" "BudgetEmail=${BUDGET_EMAIL:?set BUDGET_EMAIL}")
  [ -n "${ALERT_EMAIL:-}" ] && params+=("AlertEmail=$ALERT_EMAIL")
  aws cloudformation deploy --stack-name "$STACK" --template-file "$STATE/packaged.yaml" \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND --no-fail-on-empty-changeset \
    --parameter-overrides "${params[@]}" "${extra[@]}"
  cmd_outputs
}

cmd_outputs() {
  aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs" --output table
}

cmd_images() {
  local registry; registry="$(account).dkr.ecr.$AWS_REGION.amazonaws.com"
  aws ecr get-login-password | docker login --username AWS --password-stdin "$registry" >/dev/null
  local build=(docker build --provenance=false --platform "$PLATFORM" --build-arg "BASE_IMAGE=${BASE_IMAGE:-python:3.12-slim}")
  [ -n "${EXTRA_CA_CERT:-}" ] && build+=(--secret "id=extra_ca,src=$EXTRA_CA_CERT")
  local target repo
  for pair in "lambda:LambdaRepositoryUri" "training:TrainingRepositoryUri" "scoring:EcrRepositoryUri"; do
    target="${pair%%:*}"; repo="$(output "${pair#*:}")"
    [ -n "$repo" ] || fail "stack output ${pair#*:} missing; run '$0 stack' first"
    echo "Building $target -> $repo:$target-$TAG"
    "${build[@]}" --target "$target" -t "$repo:$target-$TAG" "$ROOT" >/dev/null
    docker push --quiet "$repo:$target-$TAG"
  done
  echo "Pushed images tagged *-$TAG"
}

cmd_data() {
  local dir="${1:-$ROOT/data/physionet}" bucket; bucket="$(output ArtifactBucketName)"
  for name in combined.parquet splits.csv; do [ -f "$dir/$name" ] || fail "$dir/$name not found"; done
  for name in combined.parquet splits.csv splits.csv.summary.json dvc.lock; do
    [ -f "$dir/$name" ] && aws s3 cp --only-show-errors "$dir/$name" "s3://$bucket/data/physionet/$name"
  done
  echo "Uploaded to s3://$bucket/data/physionet/ (encrypted with the stack's KMS key)"
}

cmd_train() {
  "$PYTHON" "$ROOT/scripts/sagemaker_train.py" --stack "$STACK" --region "$AWS_REGION" \
    --image-tag "training-$TAG" --wait "$@"
}

cmd_mlflow_ui() {
  aws sagemaker create-presigned-mlflow-app-url --arn "$(output MlflowAppArn)" \
    --query AuthorizedUrl --output text
}

cmd_promote() {
  [ $# -eq 2 ] || fail "usage: $0 promote VERSION REVIEWER"
  local real=(--require-real-data); [ "${ALLOW_SYNTHETIC:-}" = 1 ] && real=()
  "$PYTHON" -m sepsis_warning promote --version "$1" --reviewed-by "$2" --mlflow-uri "$(output MlflowAppArn)" "${real[@]}"
}

cmd_publish() {
  "$PYTHON" -m sepsis_warning publish --mlflow-uri "$(output MlflowAppArn)" \
    --bucket "$(output ArtifactBucketName)" --model-uri "${MODEL_URI:-models:/sepsis-warning@production}" \
    | tee "$STATE/publish.log"
  grep -o 'ModelS3Uri=[^ ]*' "$STATE/publish.log" | cut -d= -f2 > "$STATE/model_s3_uri"
  grep -o 'ModelSha256=[0-9a-f]*' "$STATE/publish.log" | cut -d= -f2 > "$STATE/model_sha256"
}

cmd_serve() {
  [ -s "$STATE/model_s3_uri" ] || fail "run '$0 publish' first"
  local image_param
  if [ "$SERVING_MODE" = lambda ]; then
    image_param="LambdaImageUri=$(output LambdaRepositoryUri):lambda-$TAG"
  else
    image_param="ScoringImageUri=$(output EcrRepositoryUri):scoring-$TAG"
  fi
  cmd_stack DeployScoringService=true "$image_param" \
    "ModelS3Uri=$(cat "$STATE/model_s3_uri")" "ModelSha256=$(cat "$STATE/model_sha256")"
}

cmd_call() {
  "$PYTHON" "$ROOT/scripts/call_scoring_url.py" --url "$(output ApiBaseUrl)" --region "$AWS_REGION" "$@"
}

cmd_pipeline() {
  local arn; arn="$(output PipelineArn)"; [ -n "$arn" ] || fail "stack has no pipeline (EnablePipeline=true)"
  local real=true; [ "${ALLOW_SYNTHETIC:-}" = 1 ] && real=false
  aws stepfunctions start-execution --state-machine-arn "$arn" --query executionArn --output text \
    --input "{\"image_tag\": \"training-$TAG\", \"git_commit\": \"$(git -C "$ROOT" rev-parse HEAD)\", \"require_real_data\": \"$real\"}"
  echo "The pipeline trains, then emails the alert address and waits for: $0 approve <token>"
}

cmd_approve() {
  [ $# -eq 1 ] || fail "usage: $0 approve TASK_TOKEN   (run 'promote' first; a different person from the trainer)"
  aws stepfunctions send-task-success --task-token "$1" --task-output '{"approved": true}'
}

cmd_reference() {
  [ -f "${1:-}" ] || fail "usage: $0 reference artifacts/monitoring/reference/monitoring_reference.json"
  local bucket; bucket="$(output ArtifactBucketName)"
  aws s3 cp --only-show-errors "$1" "s3://$bucket/monitoring/reference.json"
  cmd_stack "DriftReferenceS3Uri=s3://$bucket/monitoring/reference.json"
}

cmd_down() {
  echo "Deleting stack $STACK. The bucket, KMS key, ECR repositories, DynamoDB table and log groups are"
  echo "retained (DeletionPolicy: Retain); empty and delete them by hand when the project ends."
  aws cloudformation delete-stack --stack-name "$STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK"
}

case "${1:-}" in
  stack) shift; cmd_stack "$@" ;;
  outputs) cmd_outputs ;;
  images) cmd_images ;;
  data) shift; cmd_data "$@" ;;
  train) shift; cmd_train "$@" ;;
  mlflow-ui) cmd_mlflow_ui ;;
  promote) shift; cmd_promote "$@" ;;
  publish) cmd_publish ;;
  serve) cmd_serve ;;
  call) shift; cmd_call "$@" ;;
  pipeline) cmd_pipeline ;;
  approve) shift; cmd_approve "$@" ;;
  reference) shift; cmd_reference "$@" ;;
  down) cmd_down ;;
  *) sed -n '2,26p' "$0"; exit 2 ;;
esac
