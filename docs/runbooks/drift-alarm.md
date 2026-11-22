# Drift alarm

The daily job compares yesterday's predictions with the frozen monitoring
reference. It needs no labels, so it shows that something *changed*, not that
the model got worse.

1. **Read the result**: `s3://<bucket>/drift-results/<date>.json`
   (`status`, the three checks, and the five largest missingness shifts).
2. **By check:**
   - `max_missingness_shift`: a measurement is missing much more or less often
     than in training. Usually a feed or lab-panel change at the source.
     Confirm with the hospital side before touching the model.
   - `score_psi`: the score distribution moved. Look at missingness first; if
     inputs look the same, the patient mix may have changed.
   - `alert_rate_ratio`: staff are getting half or double the alerts expected.
     Tell the clinical owner; do not move the threshold without a review.
3. **`reference_mismatch`**: a new model was deployed without a new reference.
   Run `monitor-setup` for the deployed bundle and
   `aws_deploy.sh reference <file>`.
4. **`insufficient_data`**: fewer than 200 predictions that day; no judgement.
5. **Decide.** Options are: no action (explained change), fix the data feed,
   or retrain. Retraining goes through the pipeline
   (`aws_deploy.sh pipeline`): data checks, gates, a named approver, the frozen
   evaluation, then a canary release. If `RetrainImageTag` is set the alarm
   starts the pipeline itself, and it then waits for the approver.
6. **Performance drift** (calibration, utility, subgroup gaps) needs labels:
   run `monitor-replay` on stays whose labels have settled.

`DriftJobFailedAlarm` means the job itself failed, so nothing is being
watched: check `/aws/lambda/<env>-drift` and the reference object.
