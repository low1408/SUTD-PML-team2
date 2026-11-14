# Team Status

**Live issue status:** https://github.com/low1408/SUTD-PML-team2/issues

GitHub Issues are the source of truth for progress.
Add evidence and blockers in issue comments.
Close issues only after verification and review.
This file is an issue-link index, not a live status display.
Assign GitHub accounts manually; intended owners are listed below.
Existing issue descriptions are preserved by this script.

| Task ID | Owner | Task | Issue |
|---|---|---|---|
| E01 | Edwin | Download data and convert to Parquet | [#4](https://github.com/low1408/SUTD-PML-team2/issues/4) |
| E02 | Edwin | Version data with DVC | [#5](https://github.com/low1408/SUTD-PML-team2/issues/5) |
| E03 | Edwin | Data schema checks with Pandera | [#6](https://github.com/low1408/SUTD-PML-team2/issues/6) |
| E04 | Edwin | Hospital A / B split | [#7](https://github.com/low1408/SUTD-PML-team2/issues/7) |
| E05 | Edwin | Exploratory analysis | [#8](https://github.com/low1408/SUTD-PML-team2/issues/8) |
| E06 | Edwin | SIRS rule baseline | [#9](https://github.com/low1408/SUTD-PML-team2/issues/9) |
| E07 | Edwin | Prepare slides 4, 5, 9, 10 | [#10](https://github.com/low1408/SUTD-PML-team2/issues/10) |
| J01 | Jawad | Causal rolling features | [#11](https://github.com/low1408/SUTD-PML-team2/issues/11) |
| J02 | Jawad | Missing-value handling | [#12](https://github.com/low1408/SUTD-PML-team2/issues/12) |
| J03 | Jawad | Hours since last lab test | [#13](https://github.com/low1408/SUTD-PML-team2/issues/13) |
| J04 | Jawad | Leakage tests: corrupt hour t+1 and verify earlier features remain unchanged | [#14](https://github.com/low1408/SUTD-PML-team2/issues/14) |
| J05 | Jawad | Fast utility-score function | [#15](https://github.com/low1408/SUTD-PML-team2/issues/15) |
| J06 | Jawad | Shared feature module | [#16](https://github.com/low1408/SUTD-PML-team2/issues/16) |
| J07 | Jawad | Prepare slides 6, 7 | [#17](https://github.com/low1408/SUTD-PML-team2/issues/17) |
| K01 | Kaiyuan | Logistic regression | [#18](https://github.com/low1408/SUTD-PML-team2/issues/18) |
| K02 | Kaiyuan | Random forest | [#19](https://github.com/low1408/SUTD-PML-team2/issues/19) |
| K03 | Kaiyuan | LightGBM as the main model | [#20](https://github.com/low1408/SUTD-PML-team2/issues/20) |
| K04 | Kaiyuan | Probability calibration | [#21](https://github.com/low1408/SUTD-PML-team2/issues/21) |
| K05 | Kaiyuan | Alarm threshold and six-hour cooldown | [#22](https://github.com/low1408/SUTD-PML-team2/issues/22) |
| K06 | Kaiyuan | Fairness checks | [#23](https://github.com/low1408/SUTD-PML-team2/issues/23) |
| K07 | Kaiyuan | Prepare slides 3, 8, 11, 12 | [#24](https://github.com/low1408/SUTD-PML-team2/issues/24) |
| H01 | Josh | HMM forward-pass states | [#25](https://github.com/low1408/SUTD-PML-team2/issues/25) |
| H02 | Josh | Signature features | [#26](https://github.com/low1408/SUTD-PML-team2/issues/26) |
| H03 | Josh | With / without tests | [#27](https://github.com/low1408/SUTD-PML-team2/issues/27) |
| H04 | Josh | Optional: GMM | [#28](https://github.com/low1408/SUTD-PML-team2/issues/28) |
| H05 | Josh | Optional: GRU | [#29](https://github.com/low1408/SUTD-PML-team2/issues/29) |
| H06 | Josh | Prepare slide 14 | [#30](https://github.com/low1408/SUTD-PML-team2/issues/30) |
| L01 | leo | MLflow tracking and registry | [#31](https://github.com/low1408/SUTD-PML-team2/issues/31) |
| L02 | leo | FastAPI service | [#32](https://github.com/low1408/SUTD-PML-team2/issues/32) |
| L03 | leo | Docker image | [#33](https://github.com/low1408/SUTD-PML-team2/issues/33) |
| L04 | leo | CI/CD with GitHub Actions | [#34](https://github.com/low1408/SUTD-PML-team2/issues/34) |
| L05 | leo | Latency tests | [#35](https://github.com/low1408/SUTD-PML-team2/issues/35) |
| L06 | leo | AWS deployment design | [#36](https://github.com/low1408/SUTD-PML-team2/issues/36) |
| L07 | leo | Prepare slides 16, 17, 18 | [#37](https://github.com/low1408/SUTD-PML-team2/issues/37) |
| M01 | Mashie | Hospital B replay | [#38](https://github.com/low1408/SUTD-PML-team2/issues/38) |
| M02 | Mashie | Drift monitoring at two speeds | [#39](https://github.com/low1408/SUTD-PML-team2/issues/39) |
| M03 | Mashie | SHAP explanations | [#40](https://github.com/low1408/SUTD-PML-team2/issues/40) |
| M04 | Mashie | Model card | [#41](https://github.com/low1408/SUTD-PML-team2/issues/41) |
| M05 | Mashie | Risk register | [#42](https://github.com/low1408/SUTD-PML-team2/issues/42) |
| M06 | Mashie | Final deck assembly | [#43](https://github.com/low1408/SUTD-PML-team2/issues/43) |
| M07 | Mashie | Prepare slides 13, 15, 19, 20 | [#44](https://github.com/low1408/SUTD-PML-team2/issues/44) |

## Shared project boundaries

- Keep patient data under the Git-ignored `data/` directory. DVC metadata may be versioned; patient data must not be committed to Git.
- Preserve record-level split isolation. Use training data for fitting, calibration data for calibration, and validation data for model and threshold selection.
- Freeze model, feature, calibration, and alarm-policy decisions before Hospital B replay or held-out evaluation.
- Use only current and past observations for features; do not shift the supplied benchmark label again.
- Treat these assignments as planned work, not confirmation that the listed tools or capabilities are already implemented.

Load Data first before any work:
```python scripts/download_data.py --limit 500 --output data/physionet
python -m sepsis_warning prepare \
  --data data/physionet \
  --output data/physionet/combined.parquet
