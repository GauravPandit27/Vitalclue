# VitalCue — Contactless Breathing & Guided Recovery

Camera-only respiratory-rate detection (MediaPipe Pose, shoulder displacement), a
personal-baseline state machine, and a GenAI voice-guided paced-breathing intervention —
with the recovery outcome measured rather than assumed.

## Why the loop is the point

Camera-based respiratory rate is not the novel part. Google Fit ships it on Pixel
(accurate to within ~1 breath/min), Binah.ai sells it as an SDK, and NuraLogix received
FDA 510(k) clearance (K253650) for camera-based pulse and breathing rate in June 2026.

Every one of those products stops at showing you a number. None of them notice a
deviation on your behalf, guide you through an intervention, and then check whether it
worked. That closed loop — sense, decide, intervene, verify — is what VitalCue is for.
The academic framing is a just-in-time adaptive intervention; the recurring criticism of
that literature is that the decision rules firing these interventions are unsubstantiated
and most systems make the user self-report their own stress. A passively-sensed,
personal-baseline rule with a measured recovery outcome answers that directly.

## Run locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # optional — falls back to a scripted cue without it
python -m streamlit run app.py
```

The module form is deliberate: `streamlit run app.py` only works if Python's `Scripts`
directory is on PATH, which it often is not on Windows per-user installs. Running the app
as `python app.py` will not work either — Streamlit needs its own runtime to provide the
session, and without it you get a wall of `missing ScriptRunContext!` warnings and no page.

Needs a machine with a webcam — this will not run inside a sandboxed/headless environment.

Leave **Demo mode** on for live demos: it shortens calibration from 60s to 25s. Session
logging is **off by default** — nothing is written to disk unless you enable "Save session".

Model weights are not committed; they download to `models/` on first run. Behind a
TLS-inspecting corporate proxy the obvious ways to fetch them both fail — Python ignores
the Windows certificate store, and curl refuses to proceed when it cannot reach the
proxy root's revocation endpoint — so `core/model_store.py` works through several
transports and reports all of them if none succeed. If it still cannot reach the network,
the error names the exact URL and destination path to place the file manually.

## What's implemented

- `core/vision.py` — MediaPipe Tasks `PoseLandmarker` finds the chest; breathing is
  measured as vertical optical flow inside that ROI (not a two-point shoulder landmark).
  The pose model downloads to `models/` automatically on first run.
- `core/signal_processing.py` — displacement buffer → band-pass filter → FFT → breaths/min,
  plus breathing effort (excursion per breath) from the same spectrum
- `core/engine.py` — personal baseline + z-score state machine (CALIBRATING / NORMAL / STRESS / ESCALATE)
- `core/heart_rate.py` — rPPG pulse rate via PhysNet (see below)
- `core/models/physnet.py` — PhysNet architecture, vendored from rPPG-Toolbox
- `core/genai_agent.py` — Claude-generated paced-breathing cues for STRESS; fixed, non-generated copy for ESCALATE
- `core/database.py` — opt-in session logging plus recovery-episode outcomes
- `core/model_store.py` — first-run model download with fallbacks for proxied networks
- `app.py` — Streamlit UI wiring it all together live
- `tools/validate.py` — accuracy harness for both signals (see below)
- `tests/` — state machine, respiratory chain, rPPG signal path, chest-ROI vision, and
  end-to-end synthetic-breathing integration tests: `python -m unittest discover -s tests`

## Why the chest ROI, not the shoulder landmarks

A paced 20 brpm validation trial exposed the limit of the original signal. The median
landed near 9 brpm, and the saved trace showed why: band-passed shoulder-landmark
excursion had a standard deviation of 0.0037 (under two pixels of shoulder travel), while
postural drift at 6–8 brpm dominated the spectrum. There was no energy at 20 brpm to find.
At resting rates people breathe deeper, so landmarks work; at the elevated rates the
state machine has to detect, the diaphragm does more of the work and the shoulders barely
move.

Pose landmarks still find the chest. Breathing itself is now the mean vertical optical
flow inside that region, integrated into a displacement. Averaging a few thousand pixels
recovers sub-pixel motion that a two-point landmark, quantised to whole pixels, cannot
see — confirmed in `tests/test_vision.py` against a 0.4px synthetic breath that leaves
the landmark path flat while the ROI path locks onto 20 brpm. The live UI draws a green
box over the ROI so you can see what is being measured.

## Two signals, either of which can trigger

Rate alone misses a whole class of behaviour, and this was found the obvious way — by
trying to demo it. Breathing *hard* without breathing *fast* leaves a rate estimator
completely flat: five times the shoulder excursion at the same speed still reads the same
BPM. Anyone trying to act stressed by breathing heavily will not be detected at all.

So the processor also reports **effort**, the amplitude of the breathing component taken
from the bins around the spectral peak. It comes almost free, since the band-pass and FFT
have already run to get the rate. Measured linear in excursion to within 1% and flat
across 10–28 brpm to within 4%; the absolute value depends on build, posture and distance
from the camera, so like the rate it is only ever compared against that person's own
calibration.

The two are kept separate rather than merged into one arousal score, so that when the
system fires, *which* signal fired it is still answerable — the UI shows both z-scores
against their thresholds, and the cue says "breathing faster" or "breathing harder".

**Effort needs two guards, and the second one is not obvious.** It must be sustained for
10s, because reaching for a coffee moves the shoulders far more than any breath does. And
it is ignored entirely while the rate is *below* baseline — because deep-and-slow is the
calming pattern the intervention itself coaches. Without that guard, following the
guidance drove effort to z +8.7 and re-triggered STRESS within 19 seconds: the app fought
its own intervention, and the same flaw on the exit path would have held someone in STRESS
precisely because they were responding well. Both directions are regression-tested in
`tests/test_integration.py`.

Measured end to end against synthetic breathing, from a 14 brpm baseline:

| Behaviour | Detected | Fires on |
|---|---|---|
| Rapid shallow, 20 brpm | 11s | rate |
| Rapid shallow, 26 brpm | 11s | rate |
| Breathing 2x harder, same rate | 21s | effort |
| Breathing 4x harder, same rate | 18s | effort |
| Slow deep breathing, 7 brpm | never | — (correctly suppressed) |
| Calm, unchanged | never | — |

Effort is slower than rate by design: the 10s sustain requirement is most of the gap.

## Heart rate (rPPG)

Pulse rate comes from PhysNet, a pretrained spatio-temporal 3D CNN from
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) (NeurIPS 2023). The camera
picks up sub-perceptual colour changes in facial skin as blood volume varies with each
beat; the network maps a stack of face crops to that pulse waveform, and the rate is its
dominant frequency. Weights are `UBFC-rPPG_PhysNet_DiffNormalized.pth` — trained on
webcam recordings of seated subjects, the closest public training distribution to this
use case. No training required on our side; the checkpoint downloads on first run.

Two implementation details that matter:

**Frames are resampled onto a 30fps grid before inference.** The network learned its
temporal dynamics at 30fps and 128 frames at 30fps is the exact configuration it was
benchmarked at. A laptop webcam running MediaPipe delivers well under that, so crops are
buffered by *time* over a 4.27s window and resampled by nearest neighbour. Feeding native
frames instead would present the heartbeat to the network far slower than anything it saw
in training.

**The chunk length is not free, and inference is slower than it looks.** A forward pass
at T=128 measures around 2s on a laptop CPU, and PhysNet can be rebuilt at longer T but
the cost scales with it. That is far too slow to sit in the frame loop, so inference runs
on a worker thread: `estimate_rate()` returns the previous estimate immediately and never
blocks the caller. Capping torch to 2 threads costs nothing measurable — 2 threads was as
fast as 6 on a 14-core machine — and leaves the cores free for MediaPipe. The cost of the
short window is coarse frequency resolution (~14 bpm per FFT bin), which parabolic
interpolation and a median across chunks mostly absorb.

Be honest about what this is. In live testing the estimate wandered across roughly a
35 bpm spread while settling, with peak-dominance confidence around 20–30%. It is a
wellness signal, and camera pulse rate is precisely what NuraLogix holds FDA clearance
for, so claims here get scrutinised harder than the breathing path. Toggle it off if the
feed lags — it costs frame rate that the respiratory signal would rather have.

## Validating accuracy

There is no point claiming a number you have not measured. Both signals have a protocol,
neither needs lab equipment.

Breathing is validated against an on-screen pacer, so the pacer rate is the ground truth
and no respiration belt is needed:

```bash
python tools/validate.py breathing --subject S1 --rate 12
python tools/validate.py breathing --subject S1 --rate 20
```

Pulse needs an external reference, since nothing on screen can dictate a heart rate. A
smartwatch or oximeter is best; failing that, count the radial pulse during the recording
and the harness will prompt for it:

```bash
python tools/validate.py pulse --subject S1 --reference 68
python tools/validate.py pulse --subject S1
```

Then summarise everything recorded so far, optionally per signal:

```bash
python tools/validate.py analyze
python tools/validate.py analyze --metric pulse
```

`analyze` reports mean absolute error, bias, and 95% limits of agreement — the
Bland-Altman style figures that cleared devices in this space publish. Correlation alone
is not enough: Binah.ai's own blood-pressure report shows a diastolic correlation of
R = 0.447 while still meeting its error target, which is a neat demonstration that
correlation and agreement answer different questions.

Next tier is public datasets. BP4D+ is the useful one — RGB face video with a real
thoracic-impedance respiration reference. COHFACE and MAHNOB-HCI are the standard
comparison points. Published methods land anywhere from MAE 0.83 to 8.19 brpm depending
on dataset, so always name the dataset alongside the number.

## Deliberately not implemented

**Facial expression / emotion classification.** This was in an earlier draft and has been
removed, not deferred. EU AI Act Article 5(1)(f) prohibits AI systems that infer emotions
from biometric data in workplace and education settings — in force since February 2025,
penalties up to €35M or 7% of global turnover. The Commission's guidelines read the
medical/safety exception narrowly and state that general stress or burnout monitoring
does not qualify.

A breathing rate is a physiological measurement; "this person looks angry" is an
emotional inference. Only the former is defensible, so the face mesh and the expression
and gesture classifiers are gone.

Also not built: blink rate / PERCLOS fatigue path (the `FATIGUE` state exists in the enum
but is never entered), and escalation routing to an external alert.

## Scenario positioning

`driving` is the primary scenario. EU General Safety Regulation 2019/2144 has mandated
driver drowsiness warning on all new vehicles since July 2024 and advanced distraction
warning since July 2026, so in-cabin cameras are standard fitment and the safety purpose
is the exception the AI Act does recognise. Note that the ADDW rules require closed-loop
handling with no third-party access and prompt deletion — hence logging being opt-in.

`workplace` is retained only as a self-directed mode: the person using it is the sole
recipient and there is no reporting path to an employer. The earlier `exam` scenario has
been removed as it sits squarely inside the education prohibition.

## Design notes worth defending

**Escalation is decided before the LLM is invoked.** `engine.py` gates it with a
rule-based z-score test, and escalation copy is hardcoded per scenario. The model phrases
comfort; it never decides severity.

**The baseline standard deviation has a floor.** Consecutive readings come from
overlapping FFT windows, so they are highly correlated and `pstdev` collapses toward
measuring FFT jitter rather than real breathing variability. Without the floor, an
ordinary 2 brpm shift produces a z-score above the escalation threshold and the calming
path never runs. Calibration also outlasts one full FFT window and subsamples, so the
baseline points are less autocorrelated.

**Deviation is directional.** Only rates above baseline drive STRESS; breathing more
slowly than usual is not a stress signal.

**Hysteresis on every transition.** States have separate entry and exit thresholds plus a
minimum dwell time, so a z-score hovering near a boundary cannot flip the UI frame to
frame. Escalation on timeout only fires if the rate is *not* improving, with a hard cap
at 120s.

**The sample buffer is trimmed by timestamp, not by frame count.** Sizing a deque as
`window_seconds * fps` bakes in an assumed frame rate. MediaPipe pose on a laptop webcam
actually runs nearer 17fps than 30, so the original count-based buffer held close to a
full minute of samples — roughly triple the intended 22s window, with the latency to
match. The frequency maths was still correct because `fs` is derived from real
timestamps, but the window was not the length the comments claimed.

**Smoothing is edge-padded, not zero-padded.** This was the most serious bug in the
first draft. Shoulder-Y sits around 0.5 while the breathing excursion is about 0.01, so
smoothing with numpy's `mode="same"` padded with implicit zeros and dragged the end
samples from 0.5 toward 0 — a step forty times larger than the signal itself. That step
is pure low-frequency energy and it dominated the FFT, so **every** estimate collapsed
onto the lowest in-band bin no matter how the subject breathed. Synthetic signals at 12
and 18 brpm both reported 7.28. There is a regression test for this in
`tests/test_signal_processing.py`.

**The band-pass reaches down to 0.08 Hz.** The intervention guides people toward roughly
6 breaths/min (0.1 Hz). The original 0.15 Hz floor (9 brpm) put the target rate outside
the measurable band entirely, so the system could never confirm the person complied with
its own instruction. The floor has to sit below the target rather than on it — at exactly
0.1 Hz the rate lands on the filter's own roll-off and reads about 1.3 brpm high.

**Peak frequency is parabolically interpolated.** FFT resolution is 1/window, about
2.7 brpm at a 22s window, which is too coarse to separate a normal rate from a mildly
elevated one. Fitting a parabola through the peak bin and its neighbours recovers most of
that without lengthening the window. The filter also uses second-order sections rather
than transfer-function coefficients, since the cutoffs are a small fraction of Nyquist
where the `b, a` form loses precision.

On clean synthetic sine input with noise, the chain now reads within 0.1 brpm across
6–30 brpm. That is a correctness floor, **not** an accuracy claim — real numbers come
from the metronome harness.

## Known limitations (be upfront about these if asked)

- Needs steady, even lighting and the subject mostly still and facing the camera
- 60s calibration (25s in demo mode) before the first state reading
- Quality-gating drops noisy frames rather than trusting them, so a poorly lit or jittery
  feed shows fewer readings, not wrong ones
- Detection lags behaviour by roughly 11s on rate and 18–21s on effort. The rate figure is
  set by the 22s FFT window and is not tunable without giving up frequency resolution;
  the effort figure is mostly the deliberate 10s sustain requirement
- Accuracy on camera is still being quantified. The first landmark-based 20 brpm trial
  failed for lack of signal; re-run `python tools/validate.py breathing --subject S1 --rate 20`
  against the ROI path before quoting a number. Synthetic tests prove the logic, not the
  camera.
