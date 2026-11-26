#!/usr/bin/env python3
"""Decide what a change needs: which CI jobs to run, and which AWS stages a deploy would have to run.

    python scripts/ci_changes.py --base <sha> --head <sha>      # the files changed between two commits
    python scripts/ci_changes.py --all                          # unknown base: assume everything changed

The rules live in ci/change_rules.json (the first matching pattern decides a file's category, and each category lists the CI
jobs and deploy stages it needs). With GITHUB_OUTPUT set it writes one ``ci_<job>=true|false`` line per job and
``categories=...``; with GITHUB_STEP_SUMMARY set it writes a short table. The deploy planner imports ``plan``, so CI and
deploy always agree on what a change means. Standard library only.
"""

import argparse
import json
import os
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path

RULES = Path(__file__).resolve().parents[1] / "ci" / "change_rules.json"
ZERO = "0" * 40


def load_rules(path: Path = RULES) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def classify(path: str, rules: dict) -> str:
    for rule in rules["rules"]:
        if any(fnmatchcase(path, pattern) for pattern in rule["patterns"]):
            return rule["category"]
    return "other"


def plan(files: list[str] | None, rules: dict) -> dict:
    """``files`` None means the change is unknown: everything is needed."""
    if files is None:
        return {"known": False, "files": 0, "categories": ["all"], "ci": list(rules["all_ci_jobs"]),
                "deploy": list(rules["all_deploy_stages"])}
    categories = sorted({classify(path, rules) for path in files})
    ci, deploy = set(), set()
    for category in categories:
        ci.update(rules["categories"][category]["ci"])
        deploy.update(rules["categories"][category]["deploy"])
    return {"known": True, "files": len(files), "categories": categories,
            "ci": [j for j in rules["all_ci_jobs"] if j in ci], "deploy": [s for s in rules["all_deploy_stages"] if s in deploy]}


def changed_files(base: str, head: str, run=subprocess.run) -> list[str] | None:
    """Files changed from base to head; None when base is missing, all zeros or not in this clone."""
    if not base or base == ZERO:
        return None
    result = run(["git", "diff", "--name-only", f"{base}...{head}"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        result = run(["git", "diff", "--name-only", base, head], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return None
    return [line for line in result.stdout.splitlines() if line]


def write_outputs(result: dict, rules: dict, output: str | None, summary: str | None) -> None:
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            for job in rules["all_ci_jobs"]:
                handle.write(f"ci_{job}={'true' if job in result['ci'] else 'false'}\n")
            for stage in rules["all_deploy_stages"]:
                handle.write(f"deploy_{stage}={'true' if stage in result['deploy'] else 'false'}\n")
            handle.write(f"categories={','.join(result['categories']) or 'none'}\n")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("### What this change needs\n\n")
            handle.write(f"- Changed: {result['files']} files" + ("" if result["known"] else " (base unknown: everything runs)")
                         + f", categories: {', '.join(result['categories']) or 'none'}\n")
            handle.write(f"- CI jobs: {', '.join(result['ci']) or 'only the patient-data guard'}\n")
            handle.write(f"- After merging, a deploy would run: {', '.join(result['deploy']) or 'nothing'}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--all", action="store_true", help="unknown change: everything is needed")
    args = parser.parse_args(argv)
    rules = load_rules()
    files = None if args.all else changed_files(args.base, args.head)
    result = plan(files, rules)
    write_outputs(result, rules, os.environ.get("GITHUB_OUTPUT"), os.environ.get("GITHUB_STEP_SUMMARY"))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
