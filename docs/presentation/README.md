# Final presentation and design

| File | What it is |
|---|---|
| `sepsis-early-warning.pptx` | Final presentation, 22 slides, 16:9, with speaker notes. Each slide is a full-slide image rendered from the source below. |
| `source/deck.json`, `source/slides/*.html` | Editable slide source (one HTML file per slide, in order given by `deck.json`). |
| `assets/` | Images used by the slides: AWS Architecture Icons and three aggregate figures from the hospital comparison (PR #48). `blob-map.json` maps the asset ids in the slide HTML to these files. |
| `tools/render_slides.js`, `tools/build_pptx.py` | Rebuild the PowerPoint from the source. |
| `../design/sepsis-warning-pipeline.html` | Design page: data split, training, model selection, alert policy, ICU call contract, AWS deployment and MLOps loop. Open it in a browser. |

## Rebuild the PowerPoint after editing a slide

```bash
node docs/presentation/tools/render_slides.js /tmp/deck-render
python docs/presentation/tools/build_pptx.py /tmp/deck-render docs/presentation/sepsis-early-warning.pptx
```

`render_slides.js` needs Playwright with Chromium and network access to Google Fonts (Inter, Space Grotesk). `build_pptx.py` needs `python-pptx`, which is not a project dependency.

The slide text in the `.pptx` is part of the image, so text edits belong in `source/slides/*.html`, followed by a rebuild. The deck was authored in a Claude Slides artifact; an editable `.pptx` can also be exported from there (Share → Export) and committed in place of this file.

## Video

[video-script.md](video-script.md) is the script for a 13-minute recorded video: narration split by speaker, on-screen text, animation storyboards and the list of screen recordings to capture. `animations/alarm-fatigue.html` is a self-contained animation (SIRS rule against our model on one ward) to screen-record for the "If the model is wrong" slide.

## Still to fill in

Bracketed placeholders (`[__]`) still mark results that need the MLflow runs or new runs: the SIRS AUROC and AUPRC, the logistic regression row and the gradient boosting columns other than utility on the model comparison slide, the SHAP drivers, the drift report numbers on the monitoring slide, the fairness gaps, and the presentation date. The slides on HMM and signature features (a local study that was not part of the AWS run) were removed. Slides after "If the model is wrong" moved down by one number.

## Slide owners

| Owner | Slides |
|---|---|
| Edwin | 4 Data and split · 5 Bedside to model · 9 EDA · 10 SIRS baseline |
| Jawad | 6 Pipeline · 7 Causal features |
| Kaiyuan | 3 Objective · 8 Modelling ladder · 11 Score to alert · 12 Model comparison |
| Josh | 14 If the model is wrong (new) · 20 Governance (video) |
| Leo | 16 Building blocks · 17 Commit to production · 18 AWS deployment |
| Mashie | 13 Hospital B test · 15 Explanations · 19 Monitoring · 20 Governance |

## Attribution

AWS service icons are from the AWS Architecture Icons set, used to depict AWS services in architecture diagrams. Data: PhysioNet/CinC Challenge 2019 (Reyna et al., Crit Care Med 2020), CC BY 4.0. All figures are aggregates; no patient-level data is included.
