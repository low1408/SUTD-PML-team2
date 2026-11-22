"""Promotion gates: what a model must show on validation before it can become ``@production``.

``promote`` runs these on the version's validation summary (the
``selection_metrics.json`` logged with its run) and refuses when any fails.
Gates use validation numbers only; test-split results never decide a
promotion, so they stay an unbiased final report.

| Gate | Passes when |
|---|---|
| ``utility_above_no_model`` | utility beats the constant-prior reference by ``MIN_UTILITY_GAIN`` and its 95% interval is above zero |
| ``beats_sirs`` | utility is at least the SIRS rule's on the same validation stays under the same alert budget, plus ``MIN_SIRS_MARGIN`` |
| ``within_alert_budget`` | false alerts per non-septic patient-day are within the budget used for tuning |
| ``calibrated`` | Brier score is no worse than the constant-prior reference |
| ``real_data`` | (only with ``require_real_data``) the model was not trained on the synthetic fixture |

The margins are project choices, not clinical standards; change them here, in
review, not on the command line.
"""

MIN_UTILITY_GAIN = 0.02
MIN_SIRS_MARGIN = 0.0
REFERENCE_MODEL = "constant_prior"


class GateFailure(ValueError):
    """The version did not pass the promotion gates."""

    def __init__(self, results: list[dict]):
        failed = [r for r in results if not r["passed"]]
        super().__init__("promotion gates failed: " + "; ".join(f"{r['gate']} ({r['detail']})" for r in failed))
        self.results = results


def _result(gate: str, passed: bool, detail: str) -> dict:
    return {"gate": gate, "passed": bool(passed), "detail": detail}


def evaluate_gates(selection: dict | None, *, require_real_data: bool = False) -> list[dict]:
    """One result per gate. Missing evidence fails the gate instead of skipping it."""
    if not selection:
        return [_result("validation_summary", False, "the version has no logged validation summary")]
    rows = {row["model"]: row for row in selection.get("validation_models", [])}
    selected = rows.get(selection.get("selected_model"))
    if selected is None:
        return [_result("validation_summary", False, "the selected model is not in the validation summary")]
    results = []
    utility = selected.get("normalized_hourly_utility")
    reference = rows.get(REFERENCE_MODEL)

    if utility is None or reference is None:
        results.append(_result("utility_above_no_model", False, "utility or the constant-prior reference is missing"))
    else:
        low = selected.get("utility_ci_low")
        gain = utility - (reference.get("normalized_hourly_utility") or 0.0)
        ok = gain >= MIN_UTILITY_GAIN and low is not None and low > 0
        results.append(_result("utility_above_no_model", ok,
                               f"utility {utility:.3f}, gain {gain:.3f} (needs {MIN_UTILITY_GAIN}), "
                               f"interval low {low if low is None else round(low, 3)}"))

    sirs = (selection.get("reference_baselines") or {}).get("sirs")
    if utility is None or not sirs or sirs.get("normalized_hourly_utility") is None:
        results.append(_result("beats_sirs", False, "no SIRS result on the validation stays; retrain to record it"))
    else:
        margin = utility - sirs["normalized_hourly_utility"]
        results.append(_result("beats_sirs", margin >= MIN_SIRS_MARGIN,
                               f"utility {utility:.3f} vs SIRS {sirs['normalized_hourly_utility']:.3f}"))

    rate, budget = selected.get("false_alert_events_per_nonseptic_patient_day"), selection.get("validation_alarm_budget")
    if rate is None or budget is None:
        results.append(_result("within_alert_budget", False, "false-alert rate or budget is missing"))
    else:
        results.append(_result("within_alert_budget", rate <= budget + 1e-9,
                               f"{rate:.3f} false alerts per non-septic patient-day, budget {budget}"))

    brier, reference_brier = selected.get("brier_score"), (reference or {}).get("brier_score")
    if brier is None or reference_brier is None:
        results.append(_result("calibrated", False, "Brier score is missing"))
    else:
        results.append(_result("calibrated", brier <= reference_brier + 1e-12,
                               f"Brier {brier:.4f} vs constant prior {reference_brier:.4f}"))

    if require_real_data:
        real = selection.get("data_mode") == "physionet"
        results.append(_result("real_data", real, f"data mode is {selection.get('data_mode')!r}"))
    return results


def require_gates(selection: dict | None, *, require_real_data: bool = False) -> list[dict]:
    results = evaluate_gates(selection, require_real_data=require_real_data)
    if not all(r["passed"] for r in results):
        raise GateFailure(results)
    return results
