# ICU ward replay

A single offline page that plays several ICU stays hour by hour, the way a
care team would see them: each bed's current risk, threshold crossings,
alerts after the 6-hour cooldown, and an alert feed. A strip across the top
maps each panel to its real-world step (bedside → hospital record → scoring
API → alert → monitoring). Outcomes stay hidden unless the viewer ticks
**Show outcomes (hindsight)**, which adds the label window, the inferred
onset, false alerts per non-septic patient-day against the 0.2 budget, and
which septic beds were warned in the 12 hours before onset.

A synthetic version for slides is committed at
[`docs/design/ward-replay-synthetic.html`](design/ward-replay-synthetic.html).
Open it in a browser; it needs no server and loads nothing from the internet.

## Make a page

```bash
# Synthetic demo (safe to share)
python -m sepsis_warning demo --data data/synthetic --output artifacts/demo
python -m sepsis_warning ward --model artifacts/demo/model.joblib --data data/synthetic \
    --output artifacts/ward/ward-synthetic.html

# Real PhysioNet stays (private: keep under artifacts/, never commit or share)
python -m sepsis_warning ward --model artifacts/real/model.joblib \
    --data data/physionet/combined.parquet --split validation --output artifacts/ward/ward-validation.html

# Score through the running API instead of in-process (shows L02 working)
python -m sepsis_warning serve --model artifacts/demo/model.joblib &
python -m sepsis_warning ward --api-url http://127.0.0.1:8000 --data data/synthetic \
    --output artifacts/ward/ward-api.html
```

Options: `--beds` (default 12, at most 24), `--septic-share` (default 0.3),
`--max-hours` per stay (default 72), `--seed`, `--split`.

## Rules the command enforces

- **No test data before the freeze.** `--split` defaults to `validation`.
  `internal_test` and `external_test` (hospital B) are refused unless
  `--after-freeze` is passed, which should only happen once the model,
  features, calibration and threshold are frozen.
- **No record IDs.** Beds are named "Bed 01" and so on. A page built from
  PhysioNet stays still shows patient-level trajectories, so it carries a
  "Private" banner; keep it in the Git-ignored `artifacts/` folder.
- **Model and data must agree.** A synthetic model is refused on real data and
  the reverse.
- **No overwrite.** An existing output file is never replaced.
- **API mode sends inputs only.** The label and record IDs are never sent to
  `/score`; the label is joined back locally for hindsight.

## How to read it

- Septic stays are deliberately over-represented (30% of beds by default) so
  the demo shows alerts; the footer gives the real share in the source split.
- "Above · cooldown" means the risk is over the threshold but the bed already
  alerted within the last 6 hours, so no new alert is sent. The PhysioNet
  utility scores every hourly decision; staff would see only the alerts.
- The sparkline's dashed line is the threshold, red dots are alerts, and the
  shaded area (hindsight only) is the positive-label window, which starts
  6 hours before the inferred onset.
- This is a research replay of a benchmark label, not a clinical tool.
