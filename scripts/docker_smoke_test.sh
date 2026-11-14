#!/usr/bin/env bash
# Build the scoring image and check it end to end with synthetic data only (L03).
#
#   bash scripts/docker_smoke_test.sh [image-tag]
#
# Checks: runs as a non-root user; no data or model files inside the image;
# /health answers; /score returns valid probabilities for a synthetic stay;
# invalid input gets 422. Needs Docker, curl and python3 on the host.
# BASE_IMAGE=mirror.gcr.io/library/python:3.12-slim uses a mirror if Docker Hub rate-limits.
# EXTRA_CA_CERT=/path/ca.crt trusts a TLS-inspecting proxy during pip installs only.
set -euo pipefail

IMAGE="${1:-sepsis-warning:smoke}"
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
PORT="${SMOKE_PORT:-18000}"
WORK="$(mktemp -d)"
NAME="sepsis-smoke-$$"
cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run --rm -v "$WORK:/work" --entrypoint sh "$IMAGE" -c 'rm -rf /work/*' >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

echo "1/6 Build $IMAGE"
build_args=(--build-arg "BASE_IMAGE=$BASE_IMAGE")
# Optional proxy CA (build secret, not stored in the image).
[ -n "${EXTRA_CA_CERT:-}" ] && build_args+=(--secret "id=extra_ca,src=$EXTRA_CA_CERT")
docker build --quiet "${build_args[@]}" -t "$IMAGE" . >/dev/null

echo "2/6 Image runs as a non-root user"
uid="$(docker run --rm --entrypoint id "$IMAGE" -u)"
[ "$uid" != "0" ] || fail "container runs as root"
echo "    uid=$uid"

echo "3/6 No patient data or model files in the image"
found="$(docker run --rm --entrypoint sh "$IMAGE" -c \
  "find / -xdev \( -name '*.psv' -o -name '*.parquet' -o -name '*.joblib' -o -name '*.pkl' \) 2>/dev/null | grep -v '/site-packages/' || true")"
[ -z "$found" ] || fail "unexpected files: $found"

echo "4/6 Train a synthetic demo model inside the image (software fixture only)"
chmod 777 "$WORK"
docker run --rm -v "$WORK:/work" --entrypoint python "$IMAGE" \
  -m sepsis_warning demo --data /work/synthetic --output /work/demo >/dev/null
[ -f "$WORK/demo/model.joblib" ] || fail "demo model was not written"

echo "5/6 Start the API on 127.0.0.1:$PORT and wait for /health"
docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8000" -v "$WORK/demo:/models:ro" "$IMAGE" >/dev/null
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" | python3 -c \
  'import json,sys; h=json.load(sys.stdin); assert h["synthetic_model"] is True, h; print("    model:", h["model_name"])' \
  || { docker logs "$NAME" >&2; fail "/health did not answer"; }

echo "6/6 Score a synthetic stay and reject invalid input"
body='{"patient_id":"smoke-1","hours":[
  {"ICULOS":1,"HR":82,"MAP":80,"Temp":36.9,"Resp":16,"Age":60,"Gender":1,"HospAdmTime":-3},
  {"ICULOS":2,"HR":95,"MAP":72,"Temp":null,"Resp":22,"Age":60,"Gender":1,"HospAdmTime":-3},
  {"ICULOS":3,"HR":110,"MAP":64,"Temp":38.6,"Resp":26,"Age":60,"Gender":1,"HospAdmTime":-3}]}'
curl -fsS -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/score" | python3 -c '
import json, sys
r = json.load(sys.stdin)
assert len(r["hours"]) == 3, r
assert all(0.0 <= h["probability"] <= 1.0 for h in r["hours"]), r
print("    probabilities:", [round(h["probability"], 3) for h in r["hours"]])' || fail "/score response invalid"
code="$(curl -s -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' \
  -d '{"patient_id":"smoke-2","hours":[{"ICULOS":1,"SepsisLabel":1}]}' "http://127.0.0.1:$PORT/score")"
[ "$code" = "422" ] || fail "expected 422 for a label in the input, got $code"

echo "PASS: image $IMAGE"
