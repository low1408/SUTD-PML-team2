#!/usr/bin/env bash
# Run the scoring service locally in a production-like layout (infra/local/compose.yaml).
#
#   bash scripts/local_prod.sh up [MODEL_URI]      stage the model, build, start ALB + 2 tasks
#   bash scripts/local_prod.sh test                production checks through the load balancer
#   bash scripts/local_prod.sh deploy MODEL_URI    rolling update, one task at a time, under traffic
#   bash scripts/local_prod.sh logs | down
#
# MODEL_URI defaults to models:/sepsis-warning@production in the local MLflow store
# (MLFLOW_URI, default sqlite:///artifacts/mlflow.db). Staging copies the registered
# bundle to artifacts/local-prod/ (Git-ignored) and records its SHA-256; every task
# refuses to start unless the file matches. Synthetic or local models only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/artifacts/local-prod"
ENV_FILE="$STATE/stack.env"
PYTHON="${PYTHON:-python}"
MLFLOW_URI="${MLFLOW_URI:-sqlite:///$ROOT/artifacts/mlflow.db}"
ALB_PORT="${ALB_PORT:-8080}"
ALB="http://127.0.0.1:$ALB_PORT"
compose() { docker compose -f "$ROOT/infra/local/compose.yaml" --env-file "$ENV_FILE" "$@"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "  ok   $*"; }

stage() {  # resolve a registry URI, copy its bundle, write the env file
  local uri="$1"
  mkdir -p "$STATE"
  ALB_PORT="$ALB_PORT" "$PYTHON" - "$uri" "$MLFLOW_URI" "$STATE" "$ENV_FILE" <<'PY'
import json, os, shutil, sys, tempfile
from pathlib import Path
from sepsis_warning.registry import resolve_model
from sepsis_warning.scoring import file_sha256
uri, tracking, state, env_file = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
download = Path(tempfile.mkdtemp(dir=state))
path, sha, tags = resolve_model(uri, tracking, download)
target = state / f"v{tags['version']}"
target.mkdir(exist_ok=True)
shutil.copyfile(path, target / "model.joblib")
for name in ("selection_metrics.json", "run_metadata.json"):  # for GET /model (validation only)
    if (path.parent / name).is_file():
        shutil.copyfile(path.parent / name, target / name)
(target / "registry.json").write_text(json.dumps({
    "model_uri": uri, "version": tags["version"],
    "reviewed_by": tags.get("production_reviewed_by"), "promoted_at": tags.get("production_promoted_at")}))
for item in target.iterdir():
    os.chmod(item, 0o644)
if file_sha256(target / "model.joblib") != sha:
    sys.exit("staged bundle does not match the registry hash")
shutil.rmtree(download)
env_file.write_text(f"MODEL_DIR={target}\nMODEL_SHA256={sha}\nMODEL_VERSION={tags['version']}\n"
                    f"ALB_PORT={os.environ['ALB_PORT']}\n")
print(f"staged {uri} -> version {tags['version']}, sha256 {sha[:12]}, synthetic={tags.get('synthetic')}")
PY
}

lb() { compose exec -T alb sh -c "echo '$1' | socat stdio tcp4-connect:127.0.0.1:9999"; }

sha_of_env() { grep '^MODEL_SHA256=' "$ENV_FILE" | cut -d= -f2; }

score_body='{"patient_id":"local-1","hours":[
  {"ICULOS":1,"HR":82,"MAP":80,"Temp":36.9,"Resp":16,"Age":60,"Gender":1,"HospAdmTime":-3},
  {"ICULOS":2,"HR":95,"MAP":72,"Temp":null,"Resp":22,"Age":60,"Gender":1,"HospAdmTime":-3},
  {"ICULOS":3,"HR":110,"MAP":64,"Temp":38.6,"Resp":26,"Age":60,"Gender":1,"HospAdmTime":-3}]}'
score() { curl -fsS -m 10 -H 'Content-Type: application/json' -d "$score_body" "$ALB/score"; }

build() {  # same image as the smoke test and ECR; optional proxy CA as a build secret only
  local args=(--build-arg "BASE_IMAGE=${BASE_IMAGE:-python:3.12-slim}" --platform "${PLATFORM:-linux/amd64}")
  [ -n "${EXTRA_CA_CERT:-}" ] && args+=(--secret "id=extra_ca,src=$EXTRA_CA_CERT")
  docker build --quiet "${args[@]}" -t sepsis-warning:local "$ROOT" >/dev/null
}

cmd_up() {
  stage "${1:-models:/sepsis-warning@production}"
  echo "Building sepsis-warning:local"
  build
  compose up -d --wait
  for _ in $(seq 1 30); do curl -fsS -o /dev/null "$ALB/health" 2>/dev/null && break; sleep 0.5; done
  curl -fsS -o /dev/null "$ALB/health" || fail "load balancer has no healthy task"
  echo "Load balancer: $ALB  (try: curl $ALB/health)"
}

cmd_test() {
  [ -f "$ENV_FILE" ] || fail "run '$0 up' first"
  local sha; sha="$(sha_of_env)"
  echo "Production checks against $ALB"

  curl -fsS "$ALB/health" | "$PYTHON" -c 'import json,sys; h=json.load(sys.stdin); sys.exit(h["bundle_sha256"]!=sys.argv[1])' "$sha" \
    || fail "/health does not report the staged bundle"
  ok "/health through the load balancer reports the staged bundle ${sha:0:12}"

  score | "$PYTHON" -c 'import json,sys; r=json.load(sys.stdin); assert len(r["hours"])==3 and all(0<=h["probability"]<=1 for h in r["hours"])' \
    || fail "/score response invalid"
  ok "/score returns three valid probabilities"

  curl -fsS "$ALB/model" | "$PYTHON" -c '
import json, sys
text = sys.stdin.read(); m = json.loads(text)
assert m["selection"] and m["selection"]["selected_model"] == m["model_name"], m
assert not any(w in text for w in ("held_out", "internal_test", "external_test")), "held-out results served"
print("selected", m["model_name"], "from", len(m["selection"]["candidates"]), "candidates")' >/dev/null \
    || fail "/model does not describe the selection"
  ok "/model shows the selected model and its validation-only comparison"

  printf '{"stays":[{"patient_id":"bed-1","hours":%s},{"patient_id":"bed-2","hours":%s}]}' \
    "$("$PYTHON" -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1])["hours"]))' "$score_body")" '[{"ICULOS":1,"HR":80}]' \
    | curl -fsS -H 'Content-Type: application/json' -d @- "$ALB/predict" \
    | "$PYTHON" -c 'import json,sys; r=json.load(sys.stdin); assert [p["patient_id"] for p in r["predictions"]]==["bed-1","bed-2"] and r["predictions"][0]["hour"]==3' \
    || fail "/predict response invalid"
  ok "/predict returns the current risk for each stay in a ward batch"

  local upstreams; upstreams="$(for _ in 1 2 3 4 5 6; do curl -fsS -o /dev/null -D - -H 'Content-Type: application/json' \
    -d "$score_body" "$ALB/score" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-upstream"{print $2}'; done | sort -u | wc -l)"
  [ "$upstreams" -eq 2 ] || fail "requests reached $upstreams task(s), expected 2"
  ok "requests are spread across both tasks"

  local code; code="$(curl -s -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' \
    -d '{"patient_id":"local-2","hours":[{"ICULOS":1,"SepsisLabel":1}]}' "$ALB/score")"
  [ "$code" = "422" ] || fail "expected 422 for a label in the input, got $code"
  ok "input containing the label is rejected (422)"

  for task in api-a api-b; do
    [ -z "$(docker port "$(compose ps -q "$task")")" ] || fail "$task publishes a port; only the load balancer should"
  done
  ok "tasks have no published port (reachable only through the load balancer)"

  compose exec -T api-a python -c 'import urllib.request; urllib.request.urlopen("https://pypi.org", timeout=3)' >/dev/null 2>&1 \
    && fail "a task reached the internet"
  ok "tasks have no outbound internet (private network)"

  [ "$(compose exec -T api-a id -u)" != "0" ] || fail "task runs as root"
  compose exec -T api-a sh -c 'touch /app/x' >/dev/null 2>&1 && fail "root filesystem is writable"
  compose exec -T api-a sh -c 'touch /models/x' >/dev/null 2>&1 && fail "model volume is writable"
  ok "non-root user, read-only root filesystem and read-only model volume"

  if compose run --rm --no-deps -T api-a --model /models/model.joblib --model-sha256 "$(printf '0%.0s' {1..64})" >/dev/null 2>&1; then
    fail "a task started with the wrong bundle hash"
  fi
  ok "a task refuses to start when the bundle hash does not match"

  compose stop api-a >/dev/null 2>&1
  local failures=0
  for _ in $(seq 1 10); do score >/dev/null 2>&1 || failures=$((failures + 1)); done
  compose start api-a >/dev/null 2>&1
  compose up -d --wait api-a >/dev/null 2>&1
  [ "$failures" -eq 0 ] || fail "$failures of 10 requests failed with one task down"
  ok "with one task stopped, 10 of 10 requests still succeed"

  echo "Latency through the load balancer (synthetic stays):"
  "$PYTHON" "$ROOT/scripts/latency.py" --url "$ALB" --hours 24 72 --concurrency 1 4 --requests 50 \
    --output "$STATE/latency.json" | tail -n 8
  echo "PASS"
}

cmd_deploy() {
  [ -n "${1:-}" ] || fail "usage: $0 deploy models:/sepsis-warning/<version>"
  stage "$1"
  local sha; sha="$(sha_of_env)"
  local log="$STATE/deploy-traffic.log"; : > "$log"
  ( while true; do score >/dev/null 2>&1 && echo ok >> "$log" || echo fail >> "$log"; sleep 0.2; done ) &
  local traffic=$!
  for task in api-a api-b; do  # MinimumHealthyPercent 100: replace one task at a time
    echo "Replacing $task"
    lb "set server scoring_api/$task state drain" >/dev/null   # like ALB deregistration: no new requests
    sleep 2                                                     # let in-flight requests finish
    compose up -d --no-deps --wait "$task" >/dev/null 2>&1 || fail "$task did not become healthy"
    lb "set server scoring_api/$task state ready" >/dev/null
    for _ in $(seq 1 30); do lb "show stat" | grep -q "^scoring_api,$task,.*,UP," && break; sleep 0.5; done
    lb "show stat" | grep -q "^scoring_api,$task,.*,UP," || fail "load balancer did not mark $task healthy"
  done
  sleep 2; kill "$traffic"; wait "$traffic" 2>/dev/null || true
  local total failed; total="$(wc -l < "$log")"; failed="$(grep -c fail "$log" || true)"
  curl -fsS "$ALB/health" | "$PYTHON" -c 'import json,sys; h=json.load(sys.stdin); sys.exit(h["bundle_sha256"]!=sys.argv[1])' "$sha" \
    || fail "load balancer is not serving the new bundle"
  echo "Deployed ${sha:0:12}; $failed of $total requests failed during the rollout"
  [ "$failed" -eq 0 ] || fail "requests failed during the rollout"
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  test) cmd_test ;;
  deploy) shift; cmd_deploy "$@" ;;
  logs) compose logs --tail 100 ;;
  down) compose down --remove-orphans ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
