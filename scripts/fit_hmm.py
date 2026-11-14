"""H01: fit the HMM on the TRAIN split only and save its parameters.

    python scripts/fit_hmm.py --data data/physionet/combined.parquet --splits data/physionet/splits.csv

Writes artifacts/hmm/hmm_model.json. Calibration, validation and test stays are
never read for fitting. Features for any stay come from
sepsis_warning.hmm.hmm_state_features, which only uses the forward filter.
Use --limit 500 for a quick first run.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from sepsis_warning.hmm import VITALS, fit_hmm, save_hmm, vital_observations
from sepsis_warning.splits import load_split_patients


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hmm/hmm_model.json"))
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N train stays (0 = all)")
    args = parser.parse_args()

    train = load_split_patients(args.data, args.splits)["train"]
    if args.limit:
        train = train[: args.limit]
    stays = [vital_observations(p.frame) for p in train]
    hours = sum(len(s) for s in stays)
    print(f"Fitting a {args.states}-state HMM on {len(stays)} train stays ({hours} hours), vitals {VITALS}")

    started = time.time()
    model, history = fit_hmm(stays, n_states=args.states, n_iter=args.iterations)
    seconds = time.time() - started
    print(f"{len(history)} iterations in {seconds:.0f}s, log-likelihood per hour {history[-1] / hours:.3f}")

    order = np.argsort(model.means[:, 0])
    print("state  start  stay   " + "  ".join(f"{name:>6}" for name in VITALS))
    for k in order:
        print(f"{k:5d}  {model.start[k]:5.2f}  {model.transition[k, k]:4.2f}   "
              + "  ".join(f"{m:6.1f}" for m in model.means[k]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_hmm(model, args.output, extra={
        "split": "train", "stays": len(stays), "hours": hours,
        "iterations": len(history), "log_likelihood": history,
    })
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
