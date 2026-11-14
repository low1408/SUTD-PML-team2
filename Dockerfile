# syntax=docker/dockerfile:1
# Research scoring API (L02) as a container image (L03).
#
#   docker build -t sepsis-warning:local .
#   docker run --rm -p 127.0.0.1:8000:8000 \
#       -v "$PWD/artifacts/demo:/models:ro" sepsis-warning:local
#
# The image holds code and pinned packages only. The model bundle is mounted
# read-only at run time, so no patient data or model file is in any layer.
# The API has no authentication: publish the port on 127.0.0.1 only (as
# above) unless the team agrees otherwise. Only load model bundles this project
# produced, because loading a Joblib file can run code.

# Official Python image. If Docker Hub rate-limits pulls, use the same image
# from a mirror: --build-arg BASE_IMAGE=mirror.gcr.io/library/python:3.12-slim
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Pinned versions first (cached layer), then the package with the API extra.
# Behind a TLS-inspecting proxy, pass its CA as a build secret; it is used
# only during these installs and is not kept in any layer:
#   docker build --secret id=extra_ca,src=/path/to/proxy-ca.crt ...
COPY requirements.txt ./
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    python -m pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src/sepsis_warning ./src/sepsis_warning
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    python -m pip install ".[serve]" && rm -rf /app/src

# Fixed non-root user; it owns nothing in the image.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin scorer
USER 10001:10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# 0.0.0.0 is needed inside the container so the published port reaches it;
# exposure is controlled by `-p 127.0.0.1:...` on the host.
ENTRYPOINT ["python", "-m", "sepsis_warning", "serve", "--host", "0.0.0.0", "--allow-non-local-host", "--port", "8000"]
CMD ["--model", "/models/model.joblib"]
