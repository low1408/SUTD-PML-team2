# Serving: Docker image (L03) and latency (L05)

Both use synthetic data only. The API is a research scorer on a benchmark
label; it has no authentication and is not clinically validated.

## Docker image (L03)

```bash
docker build -t sepsis-warning:local .
python -m sepsis_warning demo --data data/synthetic --output artifacts/demo   # synthetic model
docker run --rm -p 127.0.0.1:8000:8000 \
    -v "$PWD/artifacts/demo:/models:ro" sepsis-warning:local
curl http://127.0.0.1:8000/health
```

What the image guarantees:

- **Non-root.** The process runs as a fixed system user (uid 10001) that owns
  nothing in the image.
- **No data or models inside.** `.dockerignore` is an allowlist (`pyproject.toml`,
  `requirements.txt`, `README.md`, `src/sepsis_warning/`), so data, models,
  artifacts, notebooks and Git history never enter the build context. The
  model bundle is mounted read-only at run time.
- **Pinned packages.** `requirements.txt` is installed first, then the package
  with the `serve` extra.
- **Local-only by default.** Inside the container the server binds 0.0.0.0 so
  the published port reaches it; publish it on `127.0.0.1` as above. Do not
  publish it more widely without team agreement.
- **Health check.** Docker probes `GET /health` every 30 seconds.

Only mount model bundles this project produced: loading a Joblib file can run
code. To refuse an unexpected bundle, override the command:
`... sepsis-warning:local --model /models/model.joblib --model-sha256 <hash>`.

### Smoke test

```bash
bash scripts/docker_smoke_test.sh
```

It builds the image, then checks:
1. it runs as a non-root user;
2. there are no `.psv`, `.parquet`, `.joblib` or `.pkl` files in it;
3. it can train a synthetic demo model;
4. `/health` answers;
5. `/score` returns valid probabilities for a 3-hour synthetic stay;
6. an input carrying `SepsisLabel` is rejected with 422.

CI runs the same script in the `docker-image` job.

Two optional settings help on restricted networks:

- `BASE_IMAGE=mirror.gcr.io/library/python:3.12-slim` pulls the same official
  base image from a mirror when Docker Hub rate-limits anonymous pulls.
- `EXTRA_CA_CERT=/path/to/proxy-ca.crt` trusts a TLS-inspecting proxy during
  the package installs only. It is passed as a BuildKit secret, so it is not
  stored in any image layer.

The image is about 1 GB, mostly NumPy, pandas, SciPy, scikit-learn, PyArrow
and Matplotlib.

## Latency (L05)

```bash
# with the API running (locally or in Docker)
python scripts/latency.py --url http://127.0.0.1:8000 --output artifacts/latency.json
```

The script sends synthetic stays to `POST /score` and reports p50, p95, p99,
mean and max latency, throughput and failures for each stay length and
concurrency. Each request carries the full stay history, so latency includes
HTTP, validation, feature building, scoring and the alert policy. Standard
library only.

### Results, 13 Nov 2026

Setup: the Docker image above with the synthetic demo model
(`hist_gradient_boosting`, 276 features), container limited to 2 CPUs, client
on the same 4-CPU Linux machine. 200 measured requests per row after 10
warm-up requests. First request after start-up: 43 ms.

| Stay length (hours) | Concurrent clients | p50 ms | p95 ms | p99 ms | Requests/s | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1 | 41 | 67 | 116 | 21.7 | 0 |
| 24 | 4 | 246 | 377 | 407 | 15.4 | 0 |
| 24 | 8 | 497 | 643 | 864 | 15.6 | 0 |
| 72 | 1 | 41 | 72 | 121 | 21.7 | 0 |
| 72 | 4 | 245 | 347 | 410 | 15.5 | 0 |
| 72 | 8 | 514 | 652 | 883 | 15.4 | 0 |
| 336 | 1 | 49 | 75 | 121 | 18.8 | 0 |
| 336 | 4 | 280 | 374 | 475 | 13.8 | 0 |
| 336 | 8 | 564 | 680 | 941 | 13.9 | 0 |

What this shows:

- **Stay length barely matters.** A 336-hour stay (the longest in the data)
  takes about 8 ms more than a 24-hour stay at p50.
- **One request at a time.** The service scores under a lock with one worker
  (`service.py`), so concurrent requests queue: latency grows with the number
  of clients and throughput stays near 15–20 requests per second.
- **Enough for the planned use.** Hourly scoring sends one request per ICU stay
  per hour, a few hundred requests an hour for one ICU. To serve more, run
  more containers (more Fargate tasks), not more threads.

No numeric latency target has been agreed, so these are measurements, not a
pass or fail. Numbers will differ on other hardware and with the final NumPy
model.

For the full AWS layout on one machine (load balancer, two tasks, private network,
rolling deploy), see [local-production.md](local-production.md).
