# Scoring errors, throttling or slow responses

**During a release:** the canary alarms roll the release back by themselves.
Confirm in CloudFormation that the stack update rolled back, then look for the
cause before trying again.

**Outside a release:**

1. **What is failing?** CloudWatch Logs Insights on `/aws/lambda/<env>-scoring`:
   ```
   fields ts, Path, status, LatencyMs, request_id, model_version
   | filter record_type = "request" and status >= 500
   | sort ts desc | limit 50
   ```
2. **`model_load_failed` lines**: the function could not load the model. Check
   that `ModelS3Uri` exists and `ModelSha256` matches (`publish` prints both).
   The function answers 503 with `Retry-After: 60` meanwhile. Fix by
   re-running `aws_deploy.sh serve` with the right values, or roll back
   ([rollback-model.md](rollback-model.md)).
3. **409 `model_mismatch`**: the image and the model disagree (features or
   scikit-learn version). Deploy the image built from the commit that trained
   the model, or retrain.
4. **Throttles**: reserved concurrency is too low for the traffic, or the
   account limit is reached. Raise `LambdaReservedConcurrency`, or ask the
   caller to batch stays with `/predict`.
5. **Slow**: p95 above 10 s usually means very long stays in large batches.
   The service caps a request at 8,000 hours; ask the caller to send smaller
   batches. Raising `LambdaMemorySize` also raises CPU.
6. **422s rising**: the caller is sending values the service refuses (labels,
   impossible values, a unit error). The response body says which field.

Callers using `ScoringClient` fall back to the SIRS rule while the service is
down; their results are marked `"fallback": true`.
