# syntax=docker/dockerfile:1
# Container images for the sepsis early-warning research code (L03).
#
# Three targets share one base and the pinned requirements:
#
#   scoring  (default)  FastAPI scoring service for ECS Fargate or a laptop
#       docker build -t sepsis-warning:local .
#       docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/artifacts/demo:/models:ro" sepsis-warning:local
#   lambda              the same API on AWS Lambda (function URL, scale to zero)
#       docker build --target lambda --provenance=false -t sepsis-warning:lambda .
#   training            SageMaker AI training job: `train` + MLflow App logging
#       docker build --target training --provenance=false -t sepsis-warning:training .
#
# Images hold code and pinned packages only. The model bundle is mounted or
# downloaded at run time (and checked against its SHA-256), and training data
# arrives through the SageMaker data channel, so no patient data or model file
# is in any layer. Only load model bundles this project produced, because
# loading a Joblib file can run code. --provenance=false keeps a single-platform
# manifest, which Lambda and SageMaker require.

# Official Python image. If Docker Hub rate-limits pulls, use the same image
# from a mirror: --build-arg BASE_IMAGE=mirror.gcr.io/library/python:3.12-slim
ARG BASE_IMAGE=python:3.12-slim

FROM ${BASE_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
# Each target installs its own hash-locked file (requirements/<target>.txt, made by
# scripts/lock_requirements.sh) with --require-hashes, so a build is reproducible and a
# changed package is refused. Behind a TLS-inspecting proxy, pass its CA as a build
# secret; it is used only during installs and kept in no layer:
#   docker build --secret id=extra_ca,src=/path/to/proxy-ca.crt ...
COPY requirements ./requirements
COPY pyproject.toml README.md ./
COPY src/sepsis_warning ./src/sepsis_warning

# SageMaker AI training job. SageMaker runs the image with the argument
# "train"; inputs and outputs use the /opt/ml layout (see sagemaker_train.py).
FROM base AS training
# requirements/training.txt pins MLflow to the SageMaker MLflow App's version (3.16 on
# 2026-11-15; the App updates itself, so re-lock with MLFLOW_VERSION when it changes).
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    python -m pip install --require-hashes -r requirements/training.txt && \
    python -m pip install --no-deps . && rm -rf /app/src /app/requirements
ENV MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet
ENTRYPOINT ["python", "-m", "sepsis_warning.sagemaker_train"]

# AWS Lambda (container image). awslambdaric talks to the Lambda runtime API;
# the handler downloads the pinned bundle from S3 into /tmp on first use.
FROM base AS lambda
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    python -m pip install --require-hashes -r requirements/lambda.txt && \
    python -m pip install --no-deps . && rm -rf /app/src /app/requirements
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["sepsis_warning.lambda_app.handler"]

# Default: the scoring API for ECS Fargate or a laptop.
FROM base AS scoring
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    python -m pip install --require-hashes -r requirements/scoring.txt && \
    python -m pip install --no-deps . && rm -rf /app/src /app/requirements
# Fixed non-root user; it owns nothing in the image.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin scorer
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
# 0.0.0.0 is needed inside the container so the published port reaches it;
# exposure is controlled by `-p 127.0.0.1:...` on the host, or the load balancer.
# The model comes from a mounted file (below) or from S3:
#   --model-s3 s3://<bucket>/models/v3/model.joblib --model-sha256 <hash>
ENTRYPOINT ["python", "-m", "sepsis_warning", "serve", "--host", "0.0.0.0", "--allow-non-local-host", "--port", "8000", "--telemetry"]
CMD ["--model", "/models/model.joblib"]
