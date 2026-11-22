"""E03: the data rules, written down in one place with Pandera.

Hard rules reject the data. Review limits never reject anything; they only count
how many readings look implausible so a person can take a look.

Error messages and reports show column names and counts only, never a patient's
identifier or reading. Blank clinical values are normal in this data and allowed.
"""

import numpy as np
import pandas as pd

from .data import DYNAMIC, STATIC

KEY = ["hospital", "patient_id"]
MIN_CELL = 10
# Broad limits chosen by this project to prompt a look, not clinical standards.
# Readings outside them are counted and kept: review only, never a rejection.
REVIEW_LIMITS = {
    "HR": (20, 250), "O2Sat": (50, 100), "Temp": (25, 44), "SBP": (40, 280), "MAP": (20, 220),
    "DBP": (10, 180), "Resp": (4, 70), "FiO2": (0.21, 1.0), "pH": (6.7, 7.8), "PaCO2": (10, 130),
    "Glucose": (20, 800), "Lactate": (0, 30), "Potassium": (1.5, 9), "Hgb": (3, 22),
    "WBC": (0.1, 150), "Platelets": (1, 1500),
}


# Values that cannot be a real measurement in the challenge's units. The scoring service
# rejects these (often a unit or transcription error); REVIEW_LIMITS only produce warnings.
HARD_LIMITS = {
    # Deliberately far outside anything in PhysioNet 2019 (sample checked: HR 22-211, Temp 31-41,
    # O2Sat 38-100, pH 6.99-7.93), so a real record is not refused; these catch unit and typing errors.
    # FiO2 has no upper limit here: fractions and percentages both occur, so it only warns.
    "HR": (0, 500), "O2Sat": (0, 105), "SaO2": (0, 105), "Temp": (5, 60), "SBP": (0, 500),
    "MAP": (0, 500), "DBP": (0, 500), "Resp": (0, 300), "pH": (5.5, 9.0),
    "Age": (0, 130), "Gender": (0, 1), "ICULOS": (1, 100000),
}
# Everything else measured is a concentration, count or time and cannot be negative.
NON_NEGATIVE_EXCEPT = {"BaseExcess", "HospAdmTime"}
MAX_WARNINGS = 20


def range_problems(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(errors, warnings) for one stay: impossible values, and values outside the review limits."""
    errors, warnings = [], []
    for name in frame.columns:
        values = frame[name]
        if name in HARD_LIMITS:
            low, high = HARD_LIMITS[name]
        elif name not in NON_NEGATIVE_EXCEPT and name in DYNAMIC:
            low, high = 0, np.inf
        else:
            low, high = -np.inf, np.inf
        bad = values.notna() & ((values < low) | (values > high))
        for row in np.flatnonzero(bad.to_numpy())[:3]:
            errors.append(f"hours[{row}].{name}={values.iloc[row]:g} is impossible (allowed {low:g} to {high:g}); "
                          "check the unit")
        if name in REVIEW_LIMITS and not bad.any():
            low, high = REVIEW_LIMITS[name]
            odd = values.notna() & ((values < low) | (values > high))
            for row in np.flatnonzero(odd.to_numpy())[:3]:
                warnings.append(f"hours[{row}].{name}={values.iloc[row]:g} is outside the review range "
                                f"{low:g} to {high:g}")
    return errors, warnings[:MAX_WARNINGS]


def _labels_never_drop(frame: pd.DataFrame):
    if not {*KEY, "SepsisLabel"} <= set(frame):
        return True  # the missing column is reported by its own rule
    return frame.groupby(KEY, sort=False)["SepsisLabel"].diff().fillna(0) >= 0


def _hours_are_consecutive(frame: pd.DataFrame):
    if not {*KEY, "ICULOS"} <= set(frame):
        return True  # the missing column is reported by its own rule
    return frame.groupby(KEY, sort=False)["ICULOS"].diff().fillna(1) == 1


def build_schema():
    """The hard rules for hourly data with hospital and patient_id columns."""
    try:
        import pandera.pandas as pa
    except ImportError as error:
        raise ValueError('Pandera is not installed; run: python -m pip install -e ".[checks]"') from error

    numeric = pa.Check(lambda s: pd.api.types.is_numeric_dtype(s), name="numeric_type")
    finite = pa.Check(lambda s: ~np.isinf(pd.to_numeric(s, errors="coerce").fillna(0)), name="no_infinite_values")
    columns = {name: pa.Column(nullable=True, checks=[numeric, finite]) for name in DYNAMIC + STATIC}
    whole_hour = pa.Check(lambda s: (s >= 1) & (s % 1 == 0), name="whole_hour_from_1")
    columns["ICULOS"] = pa.Column(nullable=False, checks=[numeric, finite, whole_hour])
    columns["SepsisLabel"] = pa.Column(nullable=False, checks=pa.Check.isin([0, 1]))
    columns["hospital"] = pa.Column(nullable=False, checks=pa.Check.isin(["A", "B"]))
    columns["patient_id"] = pa.Column(nullable=False)
    return pa.DataFrameSchema(columns, checks=[
        pa.Check(_labels_never_drop, name="label_never_drops_back_to_0"),
        pa.Check(_hours_are_consecutive, name="hours_are_consecutive"),
    ])


def validate_hourly(frame: pd.DataFrame) -> None:
    """Raise ValueError naming each broken rule. Blank clinical values are allowed."""
    schema = build_schema()
    import pandera.errors as errors

    try:
        schema.validate(frame, lazy=True)
    except errors.SchemaErrors as error:
        lines = _summarize(error.failure_cases)
    else:
        return
    raise ValueError("Data failed the schema: " + "; ".join(lines))


def _summarize(failures: pd.DataFrame) -> list[str]:
    """Rule names and row counts only, so no reading or identifier reaches a log."""
    lines = []
    for (column, check), group in failures.groupby([failures["column"].fillna("table"), "check"], sort=True):
        if check == "column_in_dataframe":
            # Here the "failure case" is the missing column's name, not patient data.
            names = ", ".join(sorted(group["failure_case"].astype(str).unique()))
            lines.append(f"missing column {names}")
        else:
            lines.append(f"{column}: {check} ({len(group)} rows)")
    return lines


def extreme_value_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count readings outside the review limits. Nothing is rejected or changed.

    Counts from 1 to MIN_CELL - 1 are shown as "<10", so a report can't point
    to a single patient's reading.
    """
    rows = []
    for name, (low, high) in REVIEW_LIMITS.items():
        values = frame[name].dropna()
        rows.append({
            "variable": name, "review_low": low, "review_high": high, "readings": len(values),
            "below_low": _hide_small(int((values < low).sum())),
            "above_high": _hide_small(int((values > high).sum())),
        })
    return pd.DataFrame(rows)


def _hide_small(count: int):
    return f"<{MIN_CELL}" if 0 < count < MIN_CELL else count
