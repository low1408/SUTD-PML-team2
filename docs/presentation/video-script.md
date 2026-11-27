# 13-minute recorded video: script, on-screen text and animation storyboards

Speech is about 11:00 at a calm pace (about 2.2 words a second, 1,450 words). The other 2:00 is breathing room for slide changes and
screen recordings, so **trim the script rather than speed up**. Numbers come from the MLflow runs of the AWS deployment
(validation 3,051 stays, internal test 3,051, hospital B 20,000). **Check every number in the MLflow UI before you record.**
Speakers follow the slide owners in [README.md](README.md); change them freely.

## 1. Running order

| # | Segment | Speaker | Slide (source file) | On screen | Start | Length |
|---|---|---|---|---|---|---|
| 1 | Opening | Leo | `cover` | slide | 0:00 | 0:20 |
| 2 | The problem | Mashie | `problem` | slide | 0:20 | 0:55 |
| 3 | The goal and the measures | Kaiyuan | `objective` | slide | 1:15 | 0:55 |
| 4 | The data | Edwin | `data` | slide | 2:10 | 0:30 |
| 5 | No cheating with time | Jawad | `features` | slide | 2:40 | 0:20 |
| 6 | The models and the choice | Kaiyuan | `results` | slide | 3:00 | 1:20 |
| 7 | The unseen hospital | Mashie | `generalisation` | slide | 4:20 | 1:00 |
| 8 | If the model is wrong | Josh | `stakes` | slide + alarm animation | 5:20 | 1:10 |
| 9 | How we ship | Leo | `release` | slide | 6:30 | 0:55 |
| 10 | The platform on AWS | Leo | `aws` | slide | 7:25 | 0:45 |
| 11 | Proof: screen recording | Leo | none | screen recordings 1 to 4 | 8:10 | 1:25 |
| 12 | Watching after release | Mashie | `monitoring` | slide, then recording 5 | 9:35 | 0:35 |
| 13 | Limits and use | Josh | `governance` | slide | 10:10 | 0:40 |
| 14 | Lessons | Jawad, Kaiyuan, Leo | `lessons` | slide | 10:50 | 0:30 |
| 15 | Close | Edwin | `close` | slide | 11:20 | 0:10 |

Skipped in the video, kept for questions (backup slides): `hospitalflow`, `pipeline`, `ladder`, `eda`, `sirs`, `policy`,
`explain`, `architecture`.

## 2. Narration

### 1. Opening (Leo, 0:20)
> Hello, we are Team 2. Today we show you an early warning for sepsis in intensive care: the model, the numbers, and the platform that
> keeps it safe. We will also tell you, honestly, where it can fail.

### 2. The problem (Mashie, 0:55)
> Sepsis is what happens when the body's reaction to an infection starts to damage its own organs. Every hour that treatment is delayed
> raises the risk of death by about four to eight percent. The difficulty is that early on the signs are quiet: vital signs drift slowly,
> and blood tests arrive every few hours, not every hour. Hospitals do have warning rules, like the SIRS rule, but they fire so often that
> nurses learn to ignore them. So there are two problems at once. Warn early enough to act, and do not flood the ward. Everything we built
> follows from those two. Picture a nurse with six patients and a monitor that beeps all shift.

### 3. The goal and the measures (Kaiyuan, 0:55)
> Our model looks at one ICU hour at a time, and uses only data up to that hour. Our main score is the challenge's clinical utility: it
> rewards an early alert, and punishes late, missed and false ones. On top of that we set a hard limit: at most 0.2 false alerts per
> non-septic patient-day. The threshold is simply the best-utility cut-off inside that limit, with a six-hour cooldown, so one patient
> does not alarm every hour. Why not accuracy? Only about two percent of hours are positive, so a model that never alerts is ninety-eight
> percent accurate and catches nobody. So we judge by utility, by false alerts, by how many septic stays we catch early, and how early.

### 4. The data (Edwin, 0:30)
> We use the public PhysioNet challenge data: forty thousand ICU stays from two hospitals, one row per hour. Hospital A, about twenty
> thousand stays, is split by stay into train, calibration, validation and test. Hospital B, twenty thousand stays, stays untouched. It is
> our unseen hospital, and we score it once, at the end.

### 5. No cheating with time (Jawad, 0:20)
> To avoid cheating, the features for hour t use only hours up to t. A missing lab is filled with its last value, plus a flag and the
> hours since it was seen. That gives 276 features, from the same code in training and in serving.

### 6. The models and the choice (Kaiyuan, 1:20)
> We compared a ladder of models: a constant prior, logistic regression, random forest and gradient boosting. All were tuned with cross
> validation grouped by patient, so one stay never sits on both sides. Gradient boosting scored a utility of 0.383 on validation, and the
> random forest 0.379. That gap is smaller than the noise, one standard error, so our rule picks the simpler model: the random forest. It
> keeps false alerts at 0.177 per patient-day, inside our budget, and catches fifty-six percent of septic stays early. The SIRS rule raises
> about two false alerts per patient-day, ten times our budget, and held to the same budget it scores only 0.018. Calibration matters too.
> It brings the Brier score from 0.137 down to 0.020, so a threshold of five percent means a real risk. The model choice was the easy part. Most of our work went into the splits, so that the numbers can be trusted.

### 7. The unseen hospital (Mashie, 1:00)
> Now the real test: a hospital the model never saw. We scored the frozen model once on hospital A's test stays, and once on all twenty
> thousand stays of hospital B. Utility falls from 0.351 to 0.255, and early detection from fifty-four to forty-seven percent. That drop is
> expected: hospital B records far fewer blood gases and its vital signs differ. But two things held. False alerts stayed inside the
> budget, at 0.14 per day, and the SIRS rule scored below zero at hospital B. So the model still helps, but less, and any new hospital
> needs its own checks. We did not retune anything on hospital B, because then B would stop being a fair test. It is exactly the situation a
> real hospital faces: a model arrives from somewhere else.

### 8. If the model is wrong (Josh, 1:10)
> Let us be clear about what is at stake. If the model misses a case, a patient gets no warning. At hospital B we catch fewer than half of
> the stays that could be caught early, so a missed case is the common failure. If it alarms too often, nurses stop trusting it, and the
> real alerts get ignored too. Here is a ward of twenty patients for one day. The SIRS rule alarms twenty times. Our model alarms four
> times, and two of those are real. And at a new hospital performance drops, as you just saw. So this is a prompt for a clinician, never a
> decision. We limit the harm by design: five gates before promotion, a different person approves, the test data is scored once, the model
> file is checked against tampering, and drift raises an alarm. None of this replaces a silent trial in a real ward.

*(Play `animations/alarm-fatigue.html` here, about 20 seconds, while Josh speaks the middle sentences. The 20 against 4 are from the
fixed random seed in that file; say "in this illustration".)*

### 9. How we ship (Leo, 0:55)
> Now how we ship it. Every change is a pull request, and CI looks at what changed: a docs change runs one check, training code runs the
> full tests and a container build. Deploying to AWS is a manual button, and it plans its own stages from what changed. A trained model is
> only a candidate. One person promotes it, a different person approves it, and only then does the frozen evaluation run, once. The image is
> tagged by git commit and the model by version, so each can roll back on its own. Nothing retrains or redeploys by itself: even a drift alarm only emails a person. That separation is the main safety control.

### 10. The platform on AWS (Leo, 0:45)
> On AWS, GitHub Actions logs in with a short-lived token, so no keys are stored. Step Functions runs prepare, tune and train as Fargate
> tasks, on cheaper Spot capacity for the long steps. MLflow and Evidently run on ECS. The approved model is published as a write-once
> file in S3 with its hash, and a Lambda function scores each hour. A power switch stops the tools when we are idle, so nothing bills while
> we are not using it: the tools cost about ten cents an hour when on, and a training task about thirty cents an hour.

### 11. Proof: screen recordings (Leo, 1:25)
Say these over recordings 1 to 4 (section 5), about 20 seconds each:
> Let me show you the evidence.
>
> **Recording 1, MLflow experiments:** Here are our three experiments: tuning, selection and evaluation. Each uses different data, so the
> test sets never touch tuning, calibration or the threshold. The tuning run used three-fold grouped cross validation and six trials per model.
>
> **Recording 2, the registered model:** This is the registered model. The tag lists the five gates it passed. It was registered by the
> training role and approved by a different role, with the reviewer named. A hash of the model file is stored, and checked before the model is served.
>
> **Recording 3, the evaluation run:** And the evaluation run carries a frozen-evaluation time. The system refuses a second look at the
> test data, so nobody can cherry-pick. Hospital A's internal test and hospital B sit side by side, each scored once.
>
> **Recording 4, GitHub Actions:** In GitHub Actions the deploy summary prints its plan and says why each stage runs, or does not. A docs-only change skips the heavy checks.

### 12. Watching after release (Mashie, 0:35)
> After release we watch at two speeds. Fast, without labels: input drift and the alert rate, in Evidently, compared with the training data.
> Slow, once labels settle: utility and calibration. An alarm emails the owner, and a person decides about retraining. To fill the prediction log we built a replay that sends hospital B stays through the live service. Evidently needs about two hundred rows before a drift report means anything.

*(Cut to recording 5, the Evidently dashboard, for the last 10 seconds.)*

### 13. Limits and use (Josh, 0:40)
> What we do not claim. This is decision support, not a medical device. It is trained on retrospective data from two US hospitals, and the
> label is a benchmark: onset times are inferred. About half of the stays that could be caught early are still missed. Before any real
> use you would need a silent trial, local threshold tuning, a clinician review and regulatory review. We deliver a model card, a data card,
> a risk register and a runbook.

### 14. Lessons (0:30)
> **Jawad:** Accuracy misleads at two percent prevalence. Splitting by stay and scoring the test sets once kept us honest.
> **Kaiyuan:** When a simpler model is within one standard error, take the simpler model.
> **Leo:** Most of our delays were in the platform: permissions, cold starts and expiring logins. Test the whole path early.

### 15. Close (Edwin, 0:10)
> Thank you. The code is on GitHub, and we are happy to take your questions.

## 3. On-screen text, slide by slide
The text is already in the slide sources in `source/slides/`; this is what each slide must show, and what to add in the edit.

| Slide | Must show | Add in the edit |
|---|---|---|
| `cover` | title, team, date | lower third with each name when they first speak |
| `problem` | "4-8% higher risk of death per hour of delay"; two problems | highlight "two problems" with a box |
| `objective` | unit, primary metric, 0.2 limit | a small badge "97.8% accurate, catches nobody" |
| `data` | 40,336 stays, split table, hospital B held out | colour hospital B differently |
| `features` | the hour-by-hour fill table | animate the fill left to right (storyboard A2) |
| `results` | random forest row bold: 0.379, 0.817, 0.121, 0.0201, 0.177, 56.1%; SIRS † row | zoom on the bold row, then on SIRS |
| `generalisation` | 0.351 and 0.255; the three bar pairs; the reason line | bars grow in sequence (storyboard A4) |
| `stakes` | three cards (~1 in 2, 10x, 0.351 to 0.255) and the safeguard band | play the alarm animation first (A3) |
| `release` | build and test row, release row, two versions | highlight the two human steps (storyboard A5) |
| `aws` | the AWS blocks, now ECS Fargate | trace the path with an arrow in the edit |
| `monitoring` | fast and slow checks | replace the blank box with recording 5 |
| `governance` | intended use, limitations, artefacts | none |
| `lessons`, `close` | three lessons, the repo | none |

## 4. Animation storyboards
Only A3 is built; the others are simple to make in any editor from the slide itself.

**A1. One patient, one thread (about 10 s each time it returns, three times).** Frame 1: a line chart of one patient's heart rate,
rising slowly. Frame 2: a marker "onset" on the right. Frame 3: a green marker "alert" 9 hours earlier. On the problem slide, show the
line only. On the results slide, add the alert. On the stakes slide, remove the alert and show the same patient missed. Same picture each
time, so the audience recognises it.

**A2. Causal fill (features slide, 10 s).** Six hour-boxes in a row. Highlight hour 2 with the reading 80, then pull the value into hours
3 and 4 with a short arrow, then show 95 at hour 5. Never an arrow from right to left.

**A3. Alarm fatigue, two wards (built: `animations/alarm-fatigue.html`, 20 s).** Two wards of 20 beds, 24 hours in about 19 seconds. The
SIRS ward flashes orange all day, 20 alarms. The model ward shows 4, two of them green and real. To record: open the file in a browser at
1280 x 720, press R, start the screen recorder, stop after the clock reads "Hour 24 of 24".

**A4. The drop at hospital B (generalisation slide, 10 s).** The three bar pairs grow one after another: early detection, false alerts,
utility. The blue bar is hospital A and the orange one hospital B. When the false-alert bars appear, show a dashed vertical line at the
0.2 budget; both bars stay inside it.

**A5. The two human steps (release slide, 12 s).** Walk a small dot along the release row: candidate, then stop at a person icon
labelled "promotes", then stop at a different person icon labelled "approves", then the frozen evaluation, then publish. The dot cannot
pass a step until its gate lights up green.

**A6. Frozen test (optional, 8 s).** The test-set box has a padlock. A hand reaches in once and the lock closes. A second hand is
refused with the words "refused: already evaluated".

## 5. Screen recordings to capture (do this first, while AWS is on)

| # | What | Length | Notes |
|---|---|---|---|
| 1 | MLflow: the experiments list, then the tuning run | 20 s | browser zoom 125% |
| 2 | MLflow: the model registry, version 1, with the gates tag and the approver tags | 20 s | open the tag values |
| 3 | MLflow: the evaluation run with the frozen-evaluation time | 20 s | |
| 4 | GitHub Actions: a deploy run summary with the plan table | 20 s | scroll slowly |
| 5 | Evidently: the Dashboard with panels, then one report | 20 s | needs the Evidently fix deployed and 200 or more prediction rows |
| 6 | (spare) the demo workflow log, with the ward replay page | 20 s | for questions or extra time |

Order of work: deploy (`mode=auto`, `retrain=no`), power `start` with your address in `allow_cidr`, the demo workflow in traffic mode,
monitor `reference` then `report`, record, then power `stop`. Hide or blur the AWS account number, your IP address and any e-mail
address. Do not show patient-level rows.

## 6. Still open before recording
- Fill the blanks marked `[__]` on the `results` slide: AUROC and AUPRC for SIRS, and the whole logistic regression row, and the
  gradient boosting columns other than utility. They are in the MLflow selection run.
- The `monitoring`, `explain` and fairness parts of `governance` need real results. The video uses recording 5 instead of the monitoring
  blanks. Leave `explain` out of the video.
- The cover date and the "work use case" line on `lessons` are yours to fill.
- Rehearse once with a timer. The speech should be 11:00 to 11:30; cut words if it runs longer.
