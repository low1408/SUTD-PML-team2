"""Read one hourly PhysioNet record per patient and keep hospital provenance."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]
# Unit1/Unit2 are deliberately excluded: administrative codes can shortcut
# learning hospital-specific measurement practices rather than physiology.
STATIC = ["Age", "Gender", "HospAdmTime", "ICULOS"]
DYNAMIC = VITALS + LABS


@dataclass
class Patient:
    patient_id: str
    hospital: str
    frame: pd.DataFrame

    @property
    def septic(self):
        return bool(self.frame["SepsisLabel"].any())


def read_patient(path: Path, require_labels=True) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="|")
    required = set(DYNAMIC + STATIC + (["SepsisLabel"] if require_labels else []))
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: empty record")
    frame = frame.apply(pd.to_numeric, errors="raise")
    if np.isinf(frame.to_numpy()).any():
        raise ValueError(f"{path}: infinite measurement")
    if "SepsisLabel" in frame:
        labels = frame["SepsisLabel"]
        if not labels.isin([0, 1]).all() or (labels.diff().dropna() < 0).any():
            raise ValueError(f"{path}: expected binary, nondecreasing challenge labels")
    if frame["ICULOS"].isna().any() or not np.allclose(
        np.diff(frame["ICULOS"]), 1
    ):
        raise ValueError(f"{path}: expected consecutive hourly ICULOS")
    return frame.reset_index(drop=True)


def load_patients(root: Path) -> list[Patient]:
    patients = []
    for hospital in ["A", "B"]:
        directory = root / f"training_set{hospital}"
        files = sorted(directory.glob("*.psv"))
        if not files:
            raise ValueError(f"No .psv files in {directory}")
        for path in files:
            patients.append(Patient(f"{hospital}:{path.stem}", hospital, read_patient(path)))
    return patients


def split_patients(patients: list[Patient], seed: int) -> dict[str, list[Patient]]:
    """Four disjoint source splits; hospital B is never used to select a model."""
    source = [p for p in patients if p.hospital == "A"]
    external = [p for p in patients if p.hospital == "B"]
    if not source or not external:
        raise ValueError("Both source hospital A and external hospital B are required")
    if min(sum(p.septic for p in source), sum(not p.septic for p in source)) < 10:
        raise ValueError("Need at least 10 septic and 10 nonseptic patients in hospital A")

    def divide(items, fraction, offset):
        return train_test_split(
            items, test_size=fraction, random_state=seed + offset,
            stratify=[p.septic for p in items],
        )

    # Approx. 55% train, 15% calibrate, 15% tune policy, 15% internal test.
    remaining, internal = divide(source, 0.15, 0)
    remaining, validation = divide(remaining, 0.15 / 0.85, 1)
    train, calibration = divide(remaining, 0.15 / 0.70, 2)
    splits = {
        "train": train, "calibration": calibration, "validation": validation,
        "internal_test": internal, "external_test": external,
    }
    ids = [p.patient_id for cohort in splits.values() for p in cohort]
    if len(ids) != len(set(ids)):
        raise ValueError("Patient IDs overlap between splits")
    for name in ["train", "calibration", "validation"]:
        if len({p.septic for p in splits[name]}) != 2:
            raise ValueError(f"{name} needs both patient outcome classes; use more data")
    return splits


def generate_synthetic(root: Path, seed=42, source_size=160, external_size=80):
    """A deliberately simplified software fixture, NOT a physiological simulator."""
    if any(root.rglob("*.psv")):
        raise ValueError(f"{root} already contains records; choose an empty demo directory")
    rng = np.random.default_rng(seed)
    for hospital, size in [("A", source_size), ("B", external_size)]:
        directory = root / f"training_set{hospital}"
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(size):
            n = int(rng.integers(40, 73))
            t = np.arange(n)
            # Artificial 20% prevalence is chosen for quick, stable smoke tests.
            septic = i % 5 == 0
            onset = int(rng.integers(23, n - 4)) if septic else n + 100
            trend = np.clip((t - (onset - 12)) / 12, 0, 1) if septic else np.zeros(n)
            # Brief disturbances also occur in nonseptic patients.
            transient = np.exp(-((t - rng.integers(10, n)) / 3) ** 2) * rng.uniform(0, 0.6)
            severity = trend + transient
            shift = 4 if hospital == "B" else 0
            frame = pd.DataFrame(np.nan, index=t, columns=DYNAMIC + STATIC)
            measurements = {
                "HR": 79 + shift + 25 * severity + rng.normal(0, 7, n),
                "O2Sat": 97 - 2 * severity + rng.normal(0, 1, n),
                "Temp": 36.8 + 1.2 * severity + rng.normal(0, 0.3, n),
                "SBP": 123 - 27 * severity + rng.normal(0, 9, n),
                "MAP": 83 - 19 * severity + rng.normal(0, 5, n),
                "DBP": 67 - 15 * severity + rng.normal(0, 5, n),
                "Resp": 17 + 9 * severity + rng.normal(0, 2, n),
                "Lactate": 1.2 + 3 * severity + rng.normal(0, 0.3, n),
                "WBC": 7 + 6 * severity + rng.normal(0, 1.2, n),
                "Creatinine": 0.9 + 0.5 * severity + rng.normal(0, 0.15, n),
                "Platelets": 235 - 75 * severity + rng.normal(0, 20, n),
            }
            for column, values in measurements.items():
                missing_rate = (0.65 if column in LABS else 0.15) + (0.1 if hospital == "B" else 0)
                values[rng.random(n) < missing_rate] = np.nan
                frame[column] = values
            frame["Age"] = int(rng.integers(25, 90))
            frame["Gender"] = int(rng.integers(0, 2))
            frame["HospAdmTime"] = -int(rng.integers(1, 100))
            frame["ICULOS"] = t + 1
            frame["SepsisLabel"] = (t >= onset - 6).astype(int) if septic else 0
            frame.to_csv(directory / f"synthetic_{i:04d}.psv", sep="|", index=False, na_rep="NaN")
