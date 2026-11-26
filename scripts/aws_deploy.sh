#!/usr/bin/env bash
# Serverless AWS workflow for the research platform (see docs/aws-serverless.md).
#
#   bash scripts/aws_deploy.sh validate              check every template with CloudFormation and preview the stack (creates nothing)
#   bash scripts/aws_deploy.sh verify [--expect-model]  check that every deployed component exists and is healthy (read-only)
#   bash scripts/aws_deploy.sh plan                  decide the deploy stages from what changed (MODE, RETRAIN, DRY_RUN)
#   bash scripts/aws_deploy.sh retrain               start a real-data training run (copy data once, MLflow on, pipeline, not followed)
#   bash scripts/aws_deploy.sh record-state          remember which stages ran for this commit (STAGES=stack,serve,...)
#   bash scripts/aws_deploy.sh ready                 wait until the hosted tools (MLflow, Evidently UI) are running and healthy
#   bash scripts/aws_deploy.sh diagnose              print why a deploy failed: stack events, then each service's stopped tasks and their logs
#   bash scripts/aws_deploy.sh stack                 create/update the stack (storage, MLflow App, roles, budget)
#   bash scripts/aws_deploy.sh images                build and push the lambda, training and scoring images
#   bash scripts/aws_deploy.sh fetch-data            copy PhysioNet Challenge 2019 from the open-data S3 bucket into ours (server-side copy)
#   bash scripts/aws_deploy.sh data [DIR]            upload combined.parquet + splits.csv (+ summary) for training
#   bash scripts/aws_deploy.sh train [ARGS...]       SageMaker training job on spot; registers @candidate
#   bash scripts/aws_deploy.sh mlflow-ui             print a short-lived link to the MLflow App UI
#   bash scripts/aws_deploy.sh promote VERSION NAME  move @production after review
#   bash scripts/aws_deploy.sh publish               copy @production to s3://<bucket>/models/vN/
#   bash scripts/aws_deploy.sh serve                 deploy the published model on Lambda (or Fargate)
#   bash scripts/aws_deploy.sh call METHOD PATH [JSON]  signed request to the scoring API
#   bash scripts/aws_deploy.sh pipeline              start the pipeline: prepare data -> tune -> train + report -> approval -> evaluate -> publish
#                                                    (DATA_SOURCE=synthetic|physionet PREPARE=true|false LIMIT=0 TRIALS=6 WAIT=1)
#   bash scripts/aws_deploy.sh pipeline-wait [ARN]   follow an execution until it needs approval, finishes or fails
#   bash scripts/aws_deploy.sh approve TOKEN         approve a waiting pipeline (the token is in the notification email)
#   bash scripts/aws_deploy.sh reference FILE        upload a monitor-setup reference and switch on the daily drift job
#   bash scripts/aws_deploy.sh power status|start|stop   show, start or stop what bills while on (MLflow, Evidently UI, database)
#   bash scripts/aws_deploy.sh outputs
#   bash scripts/aws_deploy.sh down                  delete the stack only (data, keys and images are kept)
#   bash scripts/aws_deploy.sh destroy               plan a full teardown; CONFIRM_DESTROY=<stack> performs it
#   bash scripts/aws_deploy.sh github-oidc OWNER/REPO  one-time: OIDC provider + role for deploy.yml (needs ENVIRONMENT)
#
# ENVIRONMENT=dev|staging|prod picks infra/params/<env>.json and the stack sepsis-<env>
# (use one AWS account per environment). prod needs CONFIRM_PROD=yes.
# promote refuses synthetic models unless ALLOW_SYNTHETIC=1 (dev rehearsals).
#
# Settings (environment variables): STACK (sepsis-research), AWS_REGION (required, no default),
# SERVING_MODE (lambda | fargate), TRAINING_BACKEND (sagemaker | ecs; overrides the params file), BUDGET_USD and BUDGET_EMAIL
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
# No default region: the same scripts deploy to any account, so the region is always explicit.
: "${AWS_REGION:?set AWS_REGION, ap-southeast-1 (Singapore) or ap-southeast-2 (Sydney)}"
export AWS_REGION
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

# A new account has no service-linked role for ECS (or RDS) until something creates it, and the CloudFormation handler may not:
# the first cluster then fails with "Unable to assume the service linked role. Please verify that the ECS service linked role
# exists." AWS documents the fix: `aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com` (IAM user guide,
# "Creating a service-linked role"). Done here, once, only when the stack uses the service; an existing role is left alone.
ensure_service_linked_role() {  # ensure_service_linked_role SERVICE ROLE-NAME
  wait_for_service_linked_role() {  # ROLE-NAME: service-linked roles can take time to propagate after creation
    local attempt
    for attempt in 1 2 3 4 5 6; do
      if aws iam get-role --role-name "$1" >/dev/null 2>&1; then
        aws iam wait role-exists --role-name "$1" >/dev/null 2>&1 || true
        return 0
      fi
      [ "$attempt" = 6 ] || sleep "${SERVICE_LINKED_ROLE_RETRY_SECONDS:-10}"
    done
    return 1
  }
  wait_for_ecs_service_linked_role() {
    local attempt cluster="${ECS_ROLE_PROBE_CLUSTER:-sepsis-deploy-role-probe-$$}"
    for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
      local out
      if out="$(aws ecs create-cluster --cluster-name "$cluster" --tags key=Purpose,value=service-linked-role-probe 2>&1)"; then
        aws ecs delete-cluster --cluster "$cluster" >/dev/null 2>&1 || true
        return 0
      fi
      case "$out" in
        *"Unable to assume the service linked role"*|*"Unable to assume"*linked*"role"*) ;;
        *) echo "$out" >&2; return 1 ;;
      esac
      [ "$attempt" = 12 ] || sleep "${SERVICE_LINKED_ROLE_RETRY_SECONDS:-10}"
    done
    return 1
  }
  if aws iam get-role --role-name "$2" >/dev/null 2>&1; then
    aws iam wait role-exists --role-name "$2" >/dev/null 2>&1 || true
    # ECS and CloudFormation can briefly race IAM even after get-role succeeds.
    sleep "${SERVICE_LINKED_ROLE_PROPAGATION_SECONDS:-60}"
    if [ "$1" = ecs.amazonaws.com ]; then
      wait_for_ecs_service_linked_role || fail "ECS cannot assume $2 after propagation; ask an administrator to verify $1"
    fi
    return 0
  fi
  echo "Creating the service-linked role $2 for $1 (the account has none yet)"
  local out
  if out="$(aws iam create-service-linked-role --aws-service-name "$1" 2>&1)"; then
    :
  else
    case "$out" in *"has been taken"*|*InvalidInput*taken*) ;; *)
      echo "$out" >&2
      fail "could not create $2. Ask an administrator to run: aws iam create-service-linked-role --aws-service-name $1 (or allow iam:CreateServiceLinkedRole for the deploy role / lift a service control policy that denies it)"
      ;;
    esac
  fi
  if wait_for_service_linked_role "$2"; then
    # Leave enough time for the global IAM change to reach ECS before CloudFormation creates the cluster.
    sleep "${SERVICE_LINKED_ROLE_PROPAGATION_SECONDS:-60}"
    if [ "$1" = ecs.amazonaws.com ]; then
      wait_for_ecs_service_linked_role || fail "ECS cannot assume $2 after propagation; ask an administrator to verify $1"
    fi
    return 0
  fi
  echo "The service-linked role $2 was not visible after creation" >&2
  fail "could not verify $2. Ask an administrator to run: aws iam create-service-linked-role --aws-service-name $1"
}

# What state may the stack be in when a deploy starts? A cancelled workflow does not stop CloudFormation, so an earlier run's
# operation can still be going (deploy run 29 failed at the preview because of this). Every state is handled here:
#   none / *_COMPLETE (not a rollback) ..... go ahead
#   UPDATE_IN_PROGRESS ..................... cancel it, wait for the rollback
#   UPDATE_ROLLBACK_IN_PROGRESS / CLEANUP .. wait
#   CREATE_IN_PROGRESS / ROLLBACK_IN_PROGRESS / DELETE_IN_PROGRESS ... wait (these cannot be cancelled)
#   UPDATE_ROLLBACK_COMPLETE ............... fine, the stack is back to its last good state and can be updated
#   ROLLBACK_COMPLETE ...................... a first create failed; cmd_stack deletes the empty stack
#   UPDATE_ROLLBACK_FAILED / ROLLBACK_FAILED / DELETE_FAILED ... a person must act; the message says how
settle_stack() {
  local status
  status="$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || true)"
  case "$status" in
    UPDATE_IN_PROGRESS)
      echo "Stack $STACK is still updating from an earlier run; cancelling that update so this one can start"
      aws cloudformation cancel-update-stack --stack-name "$STACK" || true
      status=UPDATE_ROLLBACK_IN_PROGRESS ;;
  esac
  case "$status" in
    UPDATE_ROLLBACK_IN_PROGRESS|UPDATE_ROLLBACK_CLEANUP_IN_PROGRESS|UPDATE_COMPLETE_CLEANUP_IN_PROGRESS)
      echo "Waiting for $STACK to finish its earlier operation ($status)"
      aws cloudformation wait stack-update-complete --stack-name "$STACK" 2>/dev/null || true
      aws cloudformation wait stack-rollback-complete --stack-name "$STACK" 2>/dev/null || true ;;
    CREATE_IN_PROGRESS|ROLLBACK_IN_PROGRESS)
      echo "Waiting for $STACK to finish its earlier operation ($status); a create cannot be cancelled"
      aws cloudformation wait stack-create-complete --stack-name "$STACK" 2>/dev/null || true
      aws cloudformation wait stack-rollback-complete --stack-name "$STACK" 2>/dev/null || true ;;
    DELETE_IN_PROGRESS)
      echo "Waiting for $STACK to finish deleting"
      aws cloudformation wait stack-delete-complete --stack-name "$STACK" 2>/dev/null || true ;;
  esac
  status="$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || true)"
  case "$status" in
    *_IN_PROGRESS) fail "stack $STACK is still $status; wait for it to finish (or cancel it in the CloudFormation console) and run again" ;;
    UPDATE_ROLLBACK_FAILED) fail "stack $STACK is UPDATE_ROLLBACK_FAILED; in the CloudFormation console choose Continue update rollback, then run again" ;;
    ROLLBACK_FAILED|DELETE_FAILED) fail "stack $STACK is $status; run the teardown workflow (it removes what blocks the delete), then deploy again" ;;
  esac
}

# If this job is cancelled while the update runs, undo the update instead of leaving it running unattended.
cancel_running_update() {
  local status
  status="$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || true)"
  if [ "$status" = UPDATE_IN_PROGRESS ]; then
    echo "Deploy interrupted: cancelling the running update of $STACK"
    aws cloudformation cancel-update-stack --stack-name "$STACK" || true
  fi
}

cmd_stack() {
  local extra=("$@")
  settle_stack
  # A first create that failed leaves the stack in ROLLBACK_COMPLETE, which cannot be updated. It holds no
  # resources, so delete it and start again.
  local status
  status="$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || true)"
  if [ "$status" = ROLLBACK_COMPLETE ]; then
    echo "Stack $STACK is in ROLLBACK_COMPLETE (an earlier create failed); deleting the empty stack first"
    aws cloudformation delete-stack --stack-name "$STACK"
    aws cloudformation wait stack-delete-complete --stack-name "$STACK"
  fi
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
  [ -n "${TRAINING_BACKEND:-}" ] && params+=("TrainingBackend=$TRAINING_BACKEND")
  local uses_ecs=false uses_rds=false item
  [ "$SERVING_MODE" = fargate ] && uses_ecs=true
  for item in "${params[@]}" "${extra[@]}"; do
    case "$item" in TrainingBackend=ecs|MlflowBackend=ecs) uses_ecs=true ;; esac
    case "$item" in MlflowBackend=ecs) uses_rds=true ;; esac
  done
  ! $uses_ecs || ensure_service_linked_role ecs.amazonaws.com AWSServiceRoleForECS
  ! $uses_rds || ensure_service_linked_role rds.amazonaws.com AWSServiceRoleForRDS
  # The MLflow server needs its image (target "tools"), which can only be pushed once the stack and its ECR
  # repository exist. Once the image for this commit is there, every stack update passes it, so the service
  # is created by the second `stack` run and is not removed by later ones.
  local tools_repo tools_tag="tools-$TAG"
  tools_repo="$(output ToolsRepositoryUri 2>/dev/null || true)"
  if [ -n "$tools_repo" ] && aws ecr describe-images --repository-name "${tools_repo##*/}" \
      --image-ids "imageTag=$tools_tag" --query 'imageDetails[0].imageDigest' --output text >/dev/null 2>&1; then
    params+=("ToolsImageUri=$tools_repo:$tools_tag")
  fi
  [ -n "${BUDGET_USD:-}" ] && params+=("MonthlyBudgetUsd=$BUDGET_USD")
  # A budget alert needs an email address; the template refuses a budget without one.
  budget_on=false
  for param in "${params[@]}"; do case "$param" in MonthlyBudgetUsd=0|MonthlyBudgetUsd=0.*) ;; MonthlyBudgetUsd=*) budget_on=true ;; esac; done
  if $budget_on; then
    [ -n "${BUDGET_EMAIL:-}" ] || fail "this environment sets a monthly budget; set BUDGET_EMAIL (or MonthlyBudgetUsd=0)"
    params+=("BudgetEmail=$BUDGET_EMAIL")
  fi
  [ -n "${ALERT_EMAIL:-}" ] && params+=("AlertEmail=$ALERT_EMAIL")
  # While the update runs, a watcher prints why any ECS task stops and cancels the update if a service is crash-looping,
  # so a broken container fails the deploy within minutes and with its cause, not after the circuit breaker.
  local watcher=""
  if [ -z "${PREVIEW:-}" ] && ! printf '%s ' "${extra[@]}" | grep -q -- '--no-execute-changeset'; then
    "$PYTHON" "$ROOT/scripts/aws_watch_services.py" watch --stack "$STACK" --region "$AWS_REGION" &
    watcher=$!
    trap 'cancel_running_update; [ -z "$watcher" ] || kill "$watcher" 2>/dev/null || true; exit 130' INT TERM
  fi
  local deployed=0
  aws cloudformation deploy --stack-name "$STACK" --template-file "$STATE/packaged.yaml" \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND --no-fail-on-empty-changeset \
    --parameter-overrides "${params[@]}" "${extra[@]}" || deployed=1
  if [ -n "$watcher" ]; then
    kill "$watcher" 2>/dev/null || true
    wait "$watcher" 2>/dev/null || true
    trap - INT TERM
  fi
  if [ "$deployed" != 0 ]; then
    local failed_set
    failed_set="$(aws cloudformation list-change-sets --stack-name "$STACK" \
      --query "Summaries[?Status=='FAILED'] | [-1].ChangeSetName" --output text 2>/dev/null | sed 's/^None$//' || true)"
    "$PYTHON" "$ROOT/scripts/aws_stack_failures.py" --stack "$STACK" --region "$AWS_REGION" ${failed_set:+--change-set "$failed_set"} || true
    fail "stack $STACK could not be created or updated; the reasons are listed above"
  fi
  [ -n "${PREVIEW:-}" ] || cmd_outputs
}

# Gate: every hosted service that is meant to run has all its tasks running and healthy (see aws_watch_services.py).
cmd_ready() {
  "$PYTHON" "$ROOT/scripts/aws_watch_services.py" ready --stack "$STACK" --region "$AWS_REGION" ${READY_TIMEOUT:+--timeout "$READY_TIMEOUT"}
}

# Why did it fail? Everything a person would otherwise look up by hand: stack failures, then each service's state and stopped tasks.
cmd_diagnose() {
  "$PYTHON" "$ROOT/scripts/aws_stack_failures.py" --stack "$STACK" --region "$AWS_REGION" || true
  "$PYTHON" "$ROOT/scripts/aws_watch_services.py" watch --stack "$STACK" --region "$AWS_REGION" --once --no-cancel || true
}

# Check every template with CloudFormation itself, then preview the stack as a change set. CloudFormation runs its
# own checks when it builds the change set (for example "a bucket with this name already exists"), so mistakes show
# up here in seconds and nothing is created or billed. cfn-lint cannot see any of this.
cmd_validate() {
  local file
  for file in "$ROOT"/infra/cloudformation/*.yaml; do
    echo "validate-template $(basename "$file")"
    aws cloudformation validate-template --template-body "file://$file" >/dev/null \
      || fail "$(basename "$file") is not a valid CloudFormation template"
  done
  echo "Previewing stack $STACK as a change set (nothing is created)"
  local ok=0
  PREVIEW=1 cmd_stack --no-execute-changeset || ok=1
  # Leave nothing behind: drop the unexecuted change sets, and the empty stack a first preview creates.
  local name
  for name in $(aws cloudformation list-change-sets --stack-name "$STACK" \
      --query "Summaries[?starts_with(ChangeSetName, 'awscli-cloudformation-package-deploy')].ChangeSetName" --output text 2>/dev/null); do
    aws cloudformation delete-change-set --stack-name "$STACK" --change-set-name "$name" >/dev/null 2>&1 || true
  done
  if [ "$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || true)" = REVIEW_IN_PROGRESS ]; then
    aws cloudformation delete-stack --stack-name "$STACK"
    aws cloudformation wait stack-delete-complete --stack-name "$STACK" || true
  fi
  [ "$ok" = 0 ] || exit 1
  echo "OK: the templates are valid and the stack preview passed"
}

# Read-only check of the deployed components (see scripts/aws_verify.py). Add --expect-model after a full deploy.
cmd_verify() {
  "$PYTHON" "$ROOT/scripts/aws_verify.py" --stack "$STACK" --region "$AWS_REGION" --tag "$TAG" "$@"
}

cmd_outputs() {
  aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs" --output table
}

# Build and push one image kind for this commit. The previous image of the same kind is pulled first and used as the layer
# cache (the dependency layers only change when a lock file does), so a code-only commit rebuilds the last two layers.
build_one_image() {  # build_one_image KIND REPOSITORY_URI
  local target="$1" repo="$2" image_tag="$1-$TAG" repository_name="${2##*/}" previous cache=()
  if aws ecr describe-images --repository-name "$repository_name" \
    --image-ids "imageTag=$image_tag" --query 'imageDetails[0].imageDigest' --output text >/dev/null 2>&1; then
    echo "Reusing existing $target image -> $repo:$image_tag"
    return 0
  fi
  previous="$(aws ecr describe-images --repository-name "$repository_name" \
    --query "sort_by(imageDetails[?imageTags!=null && starts_with(imageTags[0], '$target-')], &imagePushedAt)[-1].imageTags[0]" \
    --output text 2>/dev/null | sed 's/^None$//' || true)"
  if [ -n "$previous" ] && docker pull --quiet "$repo:$previous" >/dev/null 2>&1; then
    echo "Building $target -> $repo:$image_tag (layers reused from $previous)"
    cache=(--cache-from "$repo:$previous")
  else
    echo "Building $target -> $repo:$image_tag (no earlier image to reuse)"
  fi
  "${build[@]}" "${cache[@]}" --target "$target" -t "$repo:$image_tag" "$ROOT" >/dev/null
  docker push --quiet "$repo:$image_tag"
}

cmd_images() {
  local registry; registry="$(account).dkr.ecr.$AWS_REGION.amazonaws.com"
  aws ecr get-login-password | docker login --username AWS --password-stdin "$registry" >/dev/null
  # BUILDKIT_INLINE_CACHE stores the cache metadata in the pushed image, which the next build pulls and reuses.
  build=(docker build --provenance=false --platform "$PLATFORM" --build-arg "BASE_IMAGE=${BASE_IMAGE:-python:3.12-slim}" --build-arg BUILDKIT_INLINE_CACHE=1)
  [ -n "${EXTRA_CA_CERT:-}" ] && build+=(--secret "id=extra_ca,src=$EXTRA_CA_CERT")
  local pair target repo kinds=() repos=() i
  for pair in "lambda:LambdaRepositoryUri" "training:TrainingRepositoryUri" "tools:ToolsRepositoryUri" "scoring:EcrRepositoryUri"; do
    target="${pair%%:*}"; repo="$(output "${pair#*:}")"
    [ -n "$repo" ] || fail "stack output ${pair#*:} missing; run '$0 stack' first"
    # The scoring image is only for the always-on Fargate serving mode; Lambda mode serves from the lambda image.
    if [ "$target" = scoring ] && [ "$SERVING_MODE" != fargate ]; then
      echo "Skipping the scoring image (SERVING_MODE=$SERVING_MODE serves from the lambda image)"
      continue
    fi
    kinds+=("$target"); repos+=("$repo")
  done
  if [ "${IMAGES_PARALLEL:-}" = 1 ]; then
    # Builds are mostly waiting for package downloads, so running them side by side shortens the step a lot.
    local logs pids=() failed=0; logs="$(mktemp -d)"
    for i in "${!kinds[@]}"; do
      ( build_one_image "${kinds[$i]}" "${repos[$i]}" ) > "$logs/${kinds[$i]}.log" 2>&1 &
      pids+=($!)
    done
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || failed=1
      echo "---- ${kinds[$i]} ----"; cat "$logs/${kinds[$i]}.log"
    done
    rm -rf "$logs"
    [ "$failed" = 0 ] || fail "an image build failed; the log of each is above"
  else
    for i in "${!kinds[@]}"; do build_one_image "${kinds[$i]}" "${repos[$i]}"; done
  fi
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

# Copy the public PhysioNet Challenge 2019 training sets into our own bucket with S3's server-side copy: no web download,
# nothing passes through this machine. The source is the AWS Open Data bucket (us-east-1, public, CC BY 4.0). Re-running
# only copies what is missing. The pipeline's `prepare` step then reads these records.
cmd_fetch_data() {
  local bucket source="s3://physionet-open/challenge-2019/1.0.0/training" set count
  bucket="$(output ArtifactBucketName)"; [ -n "$bucket" ] || fail "no ArtifactBucketName output; deploy the stack first"
  # The data never changes, so this is done once: a marker records the finished copy (FORCE=1 copies again).
  if [ "${FORCE:-}" != 1 ] && aws s3api head-object --bucket "$bucket" --key raw/physionet/_SUCCESS >/dev/null 2>&1; then
    echo "The PhysioNet copy in s3://$bucket/raw/physionet/ is already complete (marker _SUCCESS); nothing to do. FORCE=1 copies again."
    return 0
  fi
  # The open-data bucket answers anonymous requests, but it refused a signed listing from this account (AccessDenied, pipeline
  # run 4), so the files come down anonymously to a scratch folder and go up with our own credentials, one hospital at a time.
  local stage; stage="$(mktemp -d)"
  trap 'rm -rf "$stage"' RETURN
  for set in A B; do
    echo "Copying $source/training_set$set/ to s3://$bucket/raw/physionet/training/training_set$set/"
    aws s3 sync "$source/training_set$set/" "$stage/training_set$set/" --no-sign-request --region us-east-1 --only-show-errors
    aws s3 sync "$stage/training_set$set/" "s3://$bucket/raw/physionet/training/training_set$set/" --region "$AWS_REGION" --only-show-errors
    rm -rf "${stage:?}/training_set$set"
    count="$(aws s3 ls "s3://$bucket/raw/physionet/training/training_set$set/" --recursive --summarize | awk '/Total Objects/ {print $3}')"
    local expected=20336; [ "$set" = B ] && expected=20000
    echo "training_set$set: $count records (expected $expected)"
    [ "$count" = "$expected" ] || fail "training_set$set has $count records, expected $expected; run fetch-data again"
  done
  printf '{"source": "%s", "training_setA": 20336, "training_setB": 20000, "copied_at": "%s"}\n' \
    "$source" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | aws s3 cp - "s3://$bucket/raw/physionet/_SUCCESS" --only-show-errors
  echo "Copy complete; marker written to s3://$bucket/raw/physionet/_SUCCESS"
}

cmd_train() {
  [ -n "$(output MlflowAppArn)" ] || fail "this stack has no SageMaker MLflow App (EnableSageMaker=false or MlflowBackend=ecs); train with the pipeline workflow (ECS), or enable SageMaker first"
  [ "$(output TrainingBackend)" = sagemaker ] || fail "training runs on ECS here: use the pipeline workflow"
  "$PYTHON" "$ROOT/scripts/sagemaker_train.py" --stack "$STACK" --region "$AWS_REGION" \
    --image-tag "training-$TAG" --wait "$@"
}

cmd_mlflow_ui() {
  [ -n "$(output MlflowAppArn)" ] || fail "this stack has no SageMaker MLflow App (EnableSageMaker=false or MlflowBackend=ecs); open the ECS MLflow with: power start --allow-cidr <your address>/32"
  aws sagemaker create-presigned-mlflow-app-url --arn "$(output MlflowAppArn)" \
    --query AuthorizedUrl --output text
}

cmd_promote() {
  [ -n "$(output MlflowAppArn)" ] || fail "this stack has no SageMaker MLflow App (EnableSageMaker=false or MlflowBackend=ecs); promote with the promote workflow, or: promote-task"
  [ $# -eq 2 ] || fail "usage: $0 promote VERSION REVIEWER"
  local real=(--require-real-data); [ "${ALLOW_SYNTHETIC:-}" = 1 ] && real=()
  "$PYTHON" -m sepsis_warning promote --version "$1" --reviewed-by "$2" --mlflow-uri "$(output MlflowAppArn)" "${real[@]}"
}

# publish: with the SageMaker MLflow App, copy the promoted version to S3 from here. With MLflow on ECS the registry is
# private and the pipeline's last step already published the approved model, so read that from S3 instead.
# ---- change-based deploy (see scripts/deploy_plan.py and docs/cicd.md)

# plan: decide the stages from what changed since each stage last ran. MODE (auto | "stack and images only" | "full deploy"),
# RETRAIN (auto | yes | no) and DRY_RUN=true are the manual options. Writes stage=true|false lines to GITHUB_OUTPUT.
cmd_plan() {
  local dry=(); [ "${DRY_RUN:-}" = true ] && dry=(--dry-run)
  "$PYTHON" "$ROOT/scripts/deploy_plan.py" plan --stack "$STACK" --region "$AWS_REGION" \
    --head "${GITHUB_SHA:-$(git -C "$ROOT" rev-parse HEAD)}" --mode "${MODE:-auto}" --retrain "${RETRAIN:-auto}" "${dry[@]}"
}

# record-state: after everything passed, remember the commit of the stages that ran (STAGES=stack,retrain,serve).
cmd_record_state() {
  "$PYTHON" "$ROOT/scripts/deploy_plan.py" record --stack "$STACK" --region "$AWS_REGION" \
    --head "${GITHUB_SHA:-$(git -C "$ROOT" rev-parse HEAD)}" --stages "${STAGES:-}" \
    --by "${GITHUB_ACTOR:-unknown}" --run "${GITHUB_RUN_ID:-}"
}

# retrain: start a real-data training run from the deploy. The PhysioNet copy is made once (a marker), MLflow is switched on if
# it is off, and the pipeline is started and not followed: it ends at your approval (promote, then approve, then deploy again).
cmd_retrain() {
  cmd_fetch_data
  cmd_power require-on --component mlflow || cmd_power start --component mlflow
  DATA_SOURCE=physionet PREPARE=true LIMIT=0 TRIALS="${TRIALS:-6}" WAIT=0 cmd_pipeline
}

# Before a full deploy spends ten minutes on images: is there a published, approved model to serve?
# Fails at once, with what to do, when the stack is not there yet or the pipeline has not published anything.
cmd_check_model() {
  local status
  status="$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || true)"
  [ -n "$status" ] || fail "stack $STACK does not exist yet: run deploy with 'stack and images only' first, then train and approve a model, then run a full deploy"
  [ -z "$(output MlflowAppArn)" ] || { echo "MLflow App is the registry: the model is checked when it is published"; return 0; }
  local allow=(); [ "${ALLOW_SYNTHETIC:-}" = 1 ] && allow=(--allow-synthetic)
  "$PYTHON" "$ROOT/scripts/aws_published_model.py" --stack "$STACK" --region "$AWS_REGION" \
    ${MODEL_VERSION:+--version "$MODEL_VERSION"} "${allow[@]}" \
    || fail "no approved published model to deploy. Run the pipeline workflow, then promote and approve, then run a full deploy"
}

cmd_publish() {
  if [ -z "$(output MlflowAppArn)" ]; then
    local allow=(); [ "${ALLOW_SYNTHETIC:-}" = 1 ] && allow=(--allow-synthetic)
    "$PYTHON" "$ROOT/scripts/aws_published_model.py" --stack "$STACK" --region "$AWS_REGION" \
      ${MODEL_VERSION:+--version "$MODEL_VERSION"} "${allow[@]}" | tee "$STATE/publish.log"
    [ "${PIPESTATUS[0]}" = 0 ] || fail "no approved published model yet"
    grep -o 'ModelS3Uri=[^ ]*' "$STATE/publish.log" | cut -d= -f2 > "$STATE/model_s3_uri"
    grep -o 'ModelSha256=[0-9a-f]*' "$STATE/publish.log" | cut -d= -f2 > "$STATE/model_sha256"
    return 0
  fi
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

# demo: replay real stays through the deployed endpoint and keep the ward page in the private bucket.
# SPLIT=validation|internal_test|external_test (held-out splits need the model's recorded frozen evaluation), BEDS, SEED.
cmd_demo() {
  "$PYTHON" "$ROOT/scripts/aws_demo.py" --stack "$STACK" --region "$AWS_REGION" \
    --split "${SPLIT:-validation}" --mode "${MODE:-ward}" --beds "${BEDS:-12}" --stays "${STAYS:-300}" --seed "${SEED:-42}"
}

# power status | start | stop [--component all|mlflow|evidently|database] [--allow-cidr x.x.x.x/32]
# Shows, starts or stops everything that bills while on (MLflow server, Evidently UI, the database).
cmd_power() {
  "$PYTHON" "$ROOT/scripts/aws_power.py" --stack "$STACK" --region "$AWS_REGION" "$@"
}

cmd_pipeline() {
  local arn; arn="$(output PipelineArn)"; [ -n "$arn" ] || fail "stack has no pipeline (EnablePipeline=true)"
  # With MLflow on ECS the server and its database must be running; they are off unless someone started them.
  if [ "$(output MlflowBackend)" = ecs ]; then
    cmd_power require-on --component mlflow || fail "start MLflow first: the power workflow with action start (or: $0 power start)"
  fi
  # DATA_SOURCE=synthetic|physionet|physionet-web  (physionet reads the copy made by `fetch-data`; physionet-web downloads over HTTPS)
  # PREPARE=true|false (false reuses the data already in S3)
  # LIMIT=stays per hospital to download (0 = all)  TRIALS=hyperparameter trials per model (0 = defaults)
  local source="${DATA_SOURCE:-synthetic}" prepare="${PREPARE:-true}" limit="${LIMIT:-0}" trials="${TRIALS:-6}" real=false
  case "$source" in synthetic) ;; physionet|physionet-web) real=true ;; *) fail "DATA_SOURCE must be synthetic, physionet or physionet-web" ;; esac
  case "$prepare" in true|false) ;; *) fail "PREPARE must be true or false" ;; esac
  case "$limit$trials" in *[!0-9]*) fail "LIMIT and TRIALS must be whole numbers" ;; esac
  local instance_type="${INSTANCE_TYPE:-ml.m5.large}"
  case "$instance_type" in ml.*) ;; *) fail "INSTANCE_TYPE must be a SageMaker instance type" ;; esac
  local input
  # FORCE_PREPARE=true prepares the data again even though the stored data has the same settings: the skip only compares the
  # source, the limit and the seed, so a change to the preparation code needs this.
  local force=false; [ "${FORCE_PREPARE:-}" = true ] && force=true
  input="$(printf '{"image_tag": "training-%s", "git_commit": "%s", "require_real_data": "%s", "prepare": "%s", "data_source": "%s", "limit": "%s", "trials": "%s", "instance_type": "%s", "force": "%s"}' \
    "$TAG" "$(git -C "$ROOT" rev-parse HEAD)" "$real" "$prepare" "$source" "$limit" "$trials" "$instance_type" "$force")"
  local execution
  execution="$(aws stepfunctions start-execution --state-machine-arn "$arn" --query executionArn --output text --input "$input")"
  echo "$execution"
  echo "Pipeline started: data=$source prepare=$prepare trials=$trials. It trains, writes the report, then emails the alert address and waits for: $0 approve <token>"
  echo "$execution" > "$STATE/pipeline_execution"
  [ "${WAIT:-}" = 1 ] && cmd_pipeline_wait "$execution"
  return 0
}

# Follow a pipeline execution until it needs a person, finishes or fails (WAIT=1 on `pipeline`, or `pipeline-wait [ARN]`).
cmd_pipeline_wait() {
  local execution="${1:-$(cat "$STATE/pipeline_execution" 2>/dev/null || true)}" status state last=""
  [ -n "$execution" ] || fail "no execution; start one with '$0 pipeline'"
  while true; do
    status="$(aws stepfunctions describe-execution --execution-arn "$execution" --query status --output text)"
    state="$(aws stepfunctions get-execution-history --execution-arn "$execution" --reverse-order --max-items 30 \
      --query "events[?stateEnteredEventDetails.name != null] | [0].stateEnteredEventDetails.name" --output text 2>/dev/null || true)"
    [ "$state" = "$last" ] || { echo "$(date +%H:%M:%S) $status: $state"; last="$state"; }
    case "$status" in SUCCEEDED|FAILED|TIMED_OUT|ABORTED) break ;; esac
    [ "$state" != WaitForApproval ] || break
    sleep 30
  done
  if [ "$status" = RUNNING ] && [ "$state" = WaitForApproval ]; then
    echo "Waiting for approval. The email with the token and the report location (s3://<bucket>/reports/<run>/) went to the alert address."
    echo "Then: $0 promote <version> \"<reviewer>\"   and   $0 approve <token>"
  elif [ "$status" != SUCCEEDED ]; then
    echo "Pipeline $status. Failed steps:"
    aws stepfunctions get-execution-history --execution-arn "$execution" \
      --query "events[?type=='TaskFailed' || type=='ExecutionFailed'].[type, taskFailedEventDetails.resource, taskFailedEventDetails.error, taskFailedEventDetails.cause, executionFailedEventDetails.cause]" --output text | head -20
    return 1
  fi
}

# promote-task: promote a model version as a one-off ECS task under the approver role (MLflow is private).
# MODEL_VERSION, REQUIRE_REAL_DATA (true|false), REVIEWED_BY and TAG are read from the environment.
cmd_promote_task() {
  local tracking; tracking="$(output MlflowTrackingUri)"
  [ -n "$tracking" ] || fail "the stack has no MLflow server on ECS (MlflowBackend=ecs); use '$0 promote' for the SageMaker MLflow App"
  cmd_power require-on --component mlflow || fail "start MLflow first: the power workflow with action start"
  case "${MODEL_VERSION:-}" in ''|*[!0-9]*) fail "MODEL_VERSION must be a whole number" ;; esac
  [ -n "${REVIEWED_BY:-}" ] || fail "REVIEWED_BY is required (the workflow passes the GitHub user)"
  "$PYTHON" "$ROOT/scripts/aws_run_task.py" --stack "$STACK" --region "$AWS_REGION" --role approver \
    --image-tag "training-$TAG" --family approver \
    --env HP_mode=promote --env "HP_mlflow_tracking_uri=$tracking" --env "HP_model_version=$MODEL_VERSION" \
    --env "HP_reviewed_by=$REVIEWED_BY" --env "HP_require_real_data=${REQUIRE_REAL_DATA:-true}" \
    --env "JOB_OUTPUT=s3://$(output ArtifactBucketName)/training-output/promote-$MODEL_VERSION-$(date +%s)/"
}

# monitor-reference: build the Evidently reference (hospital A validation, frozen model) as a one-off task in the VPC.
cmd_monitor_reference() {
  local tracking bucket; tracking="$(output MlflowTrackingUri)"; bucket="$(output ArtifactBucketName)"
  [ -n "$tracking" ] || tracking="$(output MlflowAppArn)"
  [ -n "$tracking" ] || fail "the stack has no MLflow output; deploy it first"
  [ "$(output MlflowBackend)" != ecs ] || cmd_power require-on --component mlflow || fail "start MLflow first: the power workflow with action start"
  "$PYTHON" "$ROOT/scripts/aws_run_task.py" --stack "$STACK" --region "$AWS_REGION" --role task \
    --image-tag "training-$TAG" --family monitor-reference \
    --env HP_mode=monitor_reference --env "HP_mlflow_tracking_uri=$tracking" --env "HP_publish_bucket=$bucket" \
    --env "CH_data=s3://$bucket/data/${DATA_FOLDER:-physionet}/" \
    --env "JOB_OUTPUT=s3://$bucket/training-output/monitor-reference-$(date +%s)/"
}

# monitor-report: run the Evidently drift report over the last DAYS days of the prediction log.
cmd_monitor_report() {
  local family; family="$(output EvidentlyReportFamily)"
  [ -n "$family" ] || fail "Evidently is not deployed (EnableEvidently=true, then 'stack' once the tools image exists)"
  "$PYTHON" "$ROOT/scripts/aws_run_task.py" --stack "$STACK" --region "$AWS_REGION" \
    --existing-family "$family" --container report --env "REPORT_DAYS=${DAYS:-1}"
}

# approve-run: answer the waiting pipeline from a workflow (DECISION approve|reject, EXECUTION, REASON, DECIDED_BY).
cmd_approve_run() {
  "$PYTHON" "$ROOT/scripts/aws_approval.py" --stack "$STACK" --region "$AWS_REGION" \
    --decision "${DECISION:?set DECISION to approve or reject}" --by "${DECIDED_BY:?set DECIDED_BY}" \
    ${EXECUTION:+--execution "$EXECUTION"} ${REASON:+--reason "$REASON"}
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
  echo "retained (DeletionPolicy: Retain). To remove those too: bash scripts/aws_deploy.sh destroy"
  aws cloudformation delete-stack --stack-name "$STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK"
}

cmd_github_oidc() {  # one-time per account+environment, run with admin credentials
  [ $# -eq 1 ] || fail "usage: ENVIRONMENT=dev $0 github-oidc OWNER/REPO   (the repo that runs deploy.yml)"
  [ -n "$ENVIRONMENT" ] || fail "set ENVIRONMENT=dev|staging|prod (the GitHub environment the job uses)"
  local provider_arn="arn:aws:iam::$(account):oidc-provider/token.actions.githubusercontent.com"
  local create=true
  local own_stack="github-oidc-$ENVIRONMENT"
  if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$provider_arn" >/dev/null 2>&1; then
    # Keep the provider if this stack created it; otherwise reuse the existing one.
    aws cloudformation describe-stack-resource --stack-name "$own_stack"       --logical-resource-id GitHubOidcProvider >/dev/null 2>&1 || create=false
  fi
  aws cloudformation deploy --stack-name "$own_stack"     --template-file "$ROOT/infra/cloudformation/github-oidc.yaml"     --capabilities CAPABILITY_NAMED_IAM --no-fail-on-empty-changeset     --parameter-overrides "GitHubRepo=$1" "DeployEnvironment=$ENVIRONMENT" "CreateOidcProvider=$create"
  local arn
  arn="$(aws cloudformation describe-stacks --stack-name "$own_stack"     --query "Stacks[0].Outputs[?OutputKey=='DeployRoleArn'].OutputValue | [0]" --output text)"
  echo "Set AWS_DEPLOY_ROLE_ARN=$arn on the GitHub environment '$ENVIRONMENT' of $1"
  echo "(token subject trusted: repo:$1:environment:$ENVIRONMENT)"
}

cmd_destroy() {
  local extra=(--stack "$STACK" --region "$AWS_REGION" --templates-bucket "$STACK-$(account)-$AWS_REGION-cfn")
  if [ -z "${CONFIRM_DESTROY:-}" ]; then
    "$PYTHON" "$ROOT/scripts/aws_teardown.py" "${extra[@]}"   # plan only
    echo "To delete all of this: CONFIRM_DESTROY=$STACK bash scripts/aws_deploy.sh destroy"
    return 0
  fi
  [ "${ENVIRONMENT:-}" != prod ] || [ "${CONFIRM_PROD:-}" = yes ] || fail "prod teardown needs CONFIRM_PROD=yes"
  "$PYTHON" "$ROOT/scripts/aws_teardown.py" "${extra[@]}" --yes --confirm "$CONFIRM_DESTROY"
}

case "${1:-}" in
  validate) cmd_validate ;;
  verify) shift; cmd_verify "$@" ;;
  ready) cmd_ready ;;
  diagnose) cmd_diagnose ;;
  stack) shift; cmd_stack "$@" ;;
  outputs) cmd_outputs ;;
  images) cmd_images ;;
  data) shift; cmd_data "$@" ;;
  fetch-data) cmd_fetch_data ;;
  train) shift; cmd_train "$@" ;;
  mlflow-ui) cmd_mlflow_ui ;;
  promote) shift; cmd_promote "$@" ;;
  check-model) cmd_check_model ;;
  plan) cmd_plan ;;
  record-state) cmd_record_state ;;
  retrain) cmd_retrain ;;
  publish) cmd_publish ;;
  serve) cmd_serve ;;
  call) shift; cmd_call "$@" ;;
  power) shift; cmd_power "$@" ;;
  demo) cmd_demo ;;
  monitor-reference) cmd_monitor_reference ;;
  monitor-report) cmd_monitor_report ;;
  pipeline) cmd_pipeline ;;
  pipeline-wait) shift; cmd_pipeline_wait "$@" ;;
  approve) shift; cmd_approve "$@" ;;
  approve-run) cmd_approve_run ;;
  promote-task) cmd_promote_task ;;
  reference) shift; cmd_reference "$@" ;;
  down) cmd_down ;;
  github-oidc) shift; cmd_github_oidc "$@" ;;
  destroy) cmd_destroy ;;
  *) sed -n '2,41p' "$0"; exit 2 ;;
esac
