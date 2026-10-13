"""Download a seeded subset, or all records, from the official open dataset.

    python scripts/download_data.py --limit 500 --output data/physionet

The default downloads 500 patients per hospital. --limit 0 downloads all.
The download reads data only; it does not import or execute remote code.
"""

import argparse
import hashlib
import json
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "SepsisResearchSketch/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download_one(item):
    url, destination = item
    if destination.exists():
        content = destination.read_bytes()
    else:
        content = fetch(url)
        if not content.startswith(b"HR|") or b"SepsisLabel" not in content.splitlines()[0]:
            raise ValueError(f"Unexpected PSV response from {url}")
        temporary = destination.with_suffix(".part")
        temporary.write_bytes(content)
        temporary.replace(destination)
    return {"file": str(destination), "url": url, "sha256": hashlib.sha256(content).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/physionet"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    rng = random.Random(args.seed)
    manifest = []
    for hospital in ["A", "B"]:
        url = BASE + f"training_set{hospital}/"
        index = fetch(url).decode()
        names = sorted(set(re.findall(r'href="(p\d+\.psv)"', index)))
        if not names:
            raise ValueError(f"No patient files found at {url}")
        if args.limit:
            names = sorted(rng.sample(names, min(args.limit, len(names))))
        directory = args.output / f"training_set{hospital}"
        directory.mkdir(parents=True, exist_ok=True)
        jobs = [(url + name, directory / name) for name in names]
        print(f"Downloading {len(jobs)} hospital {hospital} records", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            manifest.extend(pool.map(download_one, jobs))
    (args.output / "download_manifest.json").write_text(json.dumps({
        "source": "https://physionet.org/content/challenge-2019/1.0.0/",
        "seed": args.seed, "limit_per_hospital": args.limit, "files": manifest,
    }, indent=2))
    print("Finished. Keep raw patient records outside the submitted code archive.")


if __name__ == "__main__":
    main()
