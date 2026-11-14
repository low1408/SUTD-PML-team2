"""Measure end-to-end latency of the research scoring API (L05).

    python -m sepsis_warning serve --model artifacts/demo/model.joblib   # or the Docker image
    python scripts/latency.py --url http://127.0.0.1:8000 --output artifacts/latency.json

Sends synthetic stays only (no patient data) to ``POST /score`` and reports,
per stay length and concurrency: p50, p95, p99, mean and max latency in
milliseconds, throughput, and failures. Each request carries the full stay
history, as the stateless API expects, so latency includes request parsing,
feature building, scoring and the alert policy. The first request after the
script starts is reported separately as ``first_request_ms`` (a cold request
only if the server has just started). Standard library only.
"""

import argparse
import json
import math
import os
import platform
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VITALS = {"HR": (85, 15), "O2Sat": (97, 2), "Temp": (37.0, 0.6), "SBP": (120, 18), "MAP": (80, 12),
          "DBP": (62, 10), "Resp": (18, 4)}
LABS = {"WBC": (10, 4), "Lactate": (1.8, 1.0), "Creatinine": (1.2, 0.6), "Platelets": (210, 70),
        "Glucose": (130, 35), "BUN": (20, 10)}


def synthetic_stay(hours: int, rng: random.Random) -> list[dict]:
    """Plausible but invented hourly records: vitals most hours, labs every few hours."""
    age, gender, admit = rng.randint(25, 90), rng.randint(0, 1), -rng.uniform(0, 48)
    stay = []
    for hour in range(1, hours + 1):
        record = {"ICULOS": hour, "Age": age, "Gender": gender, "HospAdmTime": round(admit, 2)}
        for name, (mean, sd) in VITALS.items():
            record[name] = round(rng.gauss(mean, sd), 1) if rng.random() < 0.85 else None
        for name, (mean, sd) in LABS.items():
            record[name] = round(abs(rng.gauss(mean, sd)), 2) if rng.random() < 0.15 else None
        stay.append(record)
    return stay


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    rank = max(1, math.ceil(q / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def post(url: str, body: bytes, timeout: float) -> tuple[float, bool]:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            ok = response.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        ok = False
    return (time.perf_counter() - started) * 1000, ok


def run_scenario(url: str, hours: int, concurrency: int, requests: int, warmup: int,
                 timeout: float, rng: random.Random) -> dict:
    bodies = [
        json.dumps({"patient_id": f"latency-{hours}-{i}", "hours": synthetic_stay(hours, rng)}).encode()
        for i in range(8)
    ]
    for i in range(warmup):
        post(url, bodies[i % len(bodies)], timeout)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda i: post(url, bodies[i % len(bodies)], timeout), range(requests)))
    wall = time.perf_counter() - started
    latencies = sorted(ms for ms, ok in results if ok)
    failures = sum(not ok for _, ok in results)
    summary = {"hours_per_request": hours, "concurrency": concurrency, "requests": requests,
               "failures": failures, "throughput_rps": round(requests / wall, 2)}
    if latencies:
        summary.update({
            "p50_ms": round(percentile(latencies, 50), 1), "p95_ms": round(percentile(latencies, 95), 1),
            "p99_ms": round(percentile(latencies, 99), 1), "mean_ms": round(sum(latencies) / len(latencies), 1),
            "max_ms": round(latencies[-1], 1),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--hours", type=int, nargs="+", default=[24, 72, 336],
                        help="Stay lengths to send (full history per request)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--requests", type=int, default=200, help="Measured requests per scenario")
    parser.add_argument("--warmup", type=int, default=10, help="Unmeasured requests before each scenario")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file for the results")
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0 or min(args.concurrency) < 1 or min(args.hours) < 1:
        parser.error("requests, concurrency and hours must be positive; warmup must be nonnegative")

    base = args.url.rstrip("/")
    with urllib.request.urlopen(f"{base}/health", timeout=args.timeout) as response:
        health = json.load(response)
    rng = random.Random(args.seed)
    first_body = json.dumps({"patient_id": "latency-first", "hours": synthetic_stay(24, rng)}).encode()
    first_ms, first_ok = post(f"{base}/score", first_body, args.timeout)

    scenarios = [
        run_scenario(f"{base}/score", hours, concurrency, args.requests, args.warmup, args.timeout, rng)
        for hours in args.hours for concurrency in args.concurrency
    ]
    report = {
        "server": {"url": base, "model_name": health.get("model_name"),
                   "synthetic_model": health.get("synthetic_model"),
                   "feature_count": health.get("feature_count")},
        "client": {"platform": platform.platform(), "python": platform.python_version(),
                   "cpu_count": os.cpu_count()},
        "first_request_ms": round(first_ms, 1) if first_ok else None,
        "scenarios": scenarios,
        "note": "Synthetic stays; latency includes HTTP, validation, feature building, scoring and policy.",
    }
    header = f"{'hours':>5} {'conc':>4} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8} {'rps':>7} {'fail':>4}"
    print(f"Server: {report['server']['model_name']} (synthetic={report['server']['synthetic_model']}); "
          f"first request {report['first_request_ms']} ms")
    print(header)
    for row in scenarios:
        print(f"{row['hours_per_request']:>5} {row['concurrency']:>4} {row.get('p50_ms', '-'):>8} "
              f"{row.get('p95_ms', '-'):>8} {row.get('p99_ms', '-'):>8} {row.get('mean_ms', '-'):>8} "
              f"{row['throughput_rps']:>7} {row['failures']:>4}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
