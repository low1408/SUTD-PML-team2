# Runbooks and service objectives

For whoever is on call for the research scoring service. Each alarm's
description names the runbook to open.

| Alarm | Runbook |
|---|---|
| `CanaryErrorsAlarm`, `CanaryDurationAlarm`, `LambdaErrorsAlarm`, `LambdaThrottlesAlarm`, `LambdaDurationAlarm` | [scoring-errors.md](scoring-errors.md) |
| `<env>-drift`, `DriftJobFailedAlarm` | [drift-alarm.md](drift-alarm.md) |
| A bad model is live | [rollback-model.md](rollback-model.md) |

## Service objectives (proposed, not yet agreed)

These are starting points for the team and the hospital side to agree, measured
from the request metrics in the `SepsisWarning/Scoring` namespace.

| Objective | Target | Measured by |
|---|---|---|
| Availability | 99.5% of requests succeed (not 5xx) per 30 days | `ServerErrors` / request count |
| Latency | p95 under 2 s for a ward batch, excluding the first request after idle | `LatencyMs` p95 |
| Cold start | first request under 10 s | Lambda `Duration` max |
| Alert-rate band | new alerts per patient-day between 0.5x and 2x the reference | daily drift job |
| Drift review | a drift alarm is looked at within one working day | drift alarm history |

The client side has its own protection: `sepsis_warning.client.ScoringClient`
retries, opens a circuit and falls back to the SIRS rule, so a ward display is
never left without an answer.
