# Roll back a model

Use when the live model is wrong: a bad release, a failed review after the fact,
or a drift investigation that points at the model.

1. **Find the previous version.** `promote` printed it, and it is the
   `previous_production` tag on the current version:
   ```bash
   bash scripts/aws_deploy.sh mlflow-ui      # Models -> sepsis-warning -> the @production version's tags
   ```
2. **Promote it back** (the gates run again; a different person from the one who registered it):
   ```bash
   bash scripts/aws_deploy.sh promote <previous-version> "Your Name"
   ```
3. **Publish and deploy.** Publishing an already published version is a no-op;
   the deploy pins the earlier S3 object and its SHA-256.
   ```bash
   bash scripts/aws_deploy.sh publish
   bash scripts/aws_deploy.sh serve
   ```
   The stack update shifts traffic as a canary; if the old version errors too,
   it rolls back to what was running.
4. **Check.**
   ```bash
   bash scripts/aws_deploy.sh call GET /model     # model_version and bundle_sha256 must be the old ones
   ```

Faster, when seconds matter: point the `live` alias at the previous Lambda
version in the console (Lambda -> the scoring function -> Aliases -> live).
Then do steps 2-3 so the registry and the stack agree with what is running.

What does not change on a rollback: the alert ledger. Alerts already raised
stay in each stay's history, and the cooldown continues from them, so staff do
not get a burst of repeated alerts.
