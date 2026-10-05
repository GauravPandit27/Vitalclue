"""
VitalCue - Accuracy Validation Harness

Neither signal in this project should be quoted with a number until it has been measured
against something. Two protocols, both cheap:

  Breathing — the subject breathes in time with an on-screen pacer, so the pacer rate is
  the ground truth and no respiration belt is needed.

      python tools/validate.py breathing --subject S1 --rate 12
      python tools/validate.py breathing --subject S1 --rate 20

  Pulse — needs an external reference. A smartwatch, oximeter or fingertip app is best;
  failing that, count the radial pulse during the recording.

      python tools/validate.py pulse --subject S1 --reference 68
      python tools/validate.py pulse --subject S1            # prompts for a manual count

  Then summarise everything recorded so far:

      python tools/validate.py analyze
      python tools/validate.py analyze --metric pulse

Reports mean absolute error and Bland-Altman limits of agreement, which is what the
cleared devices in this space publish. Correlation on its own hides disagreement.
"""
import argparse
import csv
import math
import os
import statistics
import sys
import time

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.signal_processing import (  # noqa: E402
    RESP_HIGH_HZ,
    RESP_LOW_HZ,
    WINDOW_SECONDS,
    RespiratoryProcessor,
)
from core.vision import VisionProcessor  # noqa: E402

DEFAULT_CSV = "validation_results.csv"
TRACE_DIR = "validation_traces"
FIELDS = ["metric", "subject", "reference", "measured", "sd", "n_estimates",
          "mean_confidence", "timestamp"]

# Discard until the analysis window is full of post-onset data, otherwise the first
# estimates average in whatever the subject was doing before the trial started.
RESP_WARMUP_SECONDS = WINDOW_SECONDS + 5
PULSE_WARMUP_SECONDS = 12
INHALE_FRACTION = 0.4


# --------------------------------------------------------------------------- shared


def append_row(csv_path, metric, subject, reference, measured, sd, n, confidence):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(FIELDS)
        writer.writerow([metric, subject, f"{reference:.2f}", f"{measured:.2f}",
                         f"{sd:.2f}", n, f"{confidence:.3f}", int(time.time())])
    print(f"Appended to {csv_path}")


def report(reference, values, unit, confidence):
    """Median plus within-trial spread.

    The median alone flatters the system: it can land exactly on the target while
    individual readings wander either side, and the state machine consumes the raw
    stream, not the median. The percentage of readings within a couple of units of
    truth is the figure that predicts whether a live session behaves.
    """
    measured = statistics.median(values)
    sd = statistics.pstdev(values)
    n = len(values)
    tol = 2.0
    within = sum(1 for v in values if abs(v - reference) <= tol) / n

    print(f"\nReference {reference:.0f} {unit} -> median {measured:.1f} {unit} "
          f"(error {measured - reference:+.1f})")
    print(f"  spread    {min(values):.1f} to {max(values):.1f} {unit}, SD {sd:.2f}")
    print(f"  agreement {within:.0%} of {n} readings within +/-{tol:.0f} {unit}")
    print(f"  quality   {confidence:.0%} mean peak dominance")
    return measured, sd


def save_trace(subject, rate_bpm, trace):
    """Persist the raw shoulder signal so a bad trial can be diagnosed afterwards.

    Without this, a trial that reads 9 brpm against a 20 brpm target is unfalsifiable:
    the subject failing to keep pace and the estimator locking onto postural sway
    produce an identical summary number, and only the spectrum tells them apart.
    """
    os.makedirs(TRACE_DIR, exist_ok=True)
    path = os.path.join(TRACE_DIR, f"{subject}_{rate_bpm:g}brpm_{int(time.time())}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "resp_signal"])
        writer.writerows([f"{t:.4f}", f"{y:.6f}"] for t, y in trace)
    print(f"Raw trace saved to {path}")
    return path


def status_bar(canvas, text, seconds_left, warming_up):
    h = canvas.shape[0]
    state = "warming up" if warming_up else "recording"
    cv2.putText(canvas, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(canvas, f"{state} - {seconds_left:.0f}s left", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


# ------------------------------------------------------------------------ breathing


def draw_pacer(canvas, phase, rate_bpm, seconds_left, warming_up):
    """Expanding/contracting circle the subject breathes along with."""
    h, w = canvas.shape[:2]
    center = (w // 2, h // 2)

    if phase < INHALE_FRACTION:
        progress = phase / INHALE_FRACTION
        label = "INHALE"
    else:
        progress = 1.0 - (phase - INHALE_FRACTION) / (1.0 - INHALE_FRACTION)
        label = "EXHALE"

    min_r, max_r = 40, min(h, w) // 2 - 40
    radius = int(min_r + (max_r - min_r) * progress)

    cv2.circle(canvas, center, max_r, (60, 60, 60), 1)
    cv2.circle(canvas, center, radius, (0, 200, 120), 3)
    cv2.putText(canvas, label, (center[0] - 60, center[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
    status_bar(canvas, f"Target: {rate_bpm} breaths/min", seconds_left, warming_up)


def record_breathing(subject, rate_bpm, duration, csv_path):
    period = 60.0 / rate_bpm
    vision = VisionProcessor()
    resp = RespiratoryProcessor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available.")
        return

    print(f"Trial: subject={subject} target={rate_bpm} brpm duration={duration}s")
    print(f"Breathe with the circle. First {RESP_WARMUP_SECONDS:.0f}s are discarded.\n")

    start = time.time()
    estimates = []
    trace = []
    total = duration + RESP_WARMUP_SECONDS

    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= total:
                break
            ok, frame = cap.read()
            if not ok:
                break

            _, resp_signal, alignment = vision.process_frame(frame)
            resp.add_sample(resp_signal, time.time())
            bpm, confidence = resp.estimate_rate()

            warming_up = elapsed < RESP_WARMUP_SECONDS
            if resp_signal is not None:
                trace.append((elapsed, resp_signal))
            if not warming_up and bpm and confidence >= 0.05:
                estimates.append((bpm, confidence))

            pacer = np.zeros((480, 640, 3), dtype=np.uint8)
            draw_pacer(pacer, (elapsed % period) / period, rate_bpm, total - elapsed, warming_up)
            if alignment != "ALIGNED":
                cv2.putText(pacer, alignment, (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 160, 240), 1)

            cv2.imshow("VitalCue pacer - press q to abort", pacer)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Aborted.")
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()
        vision.close()

    if not estimates:
        print("No usable estimates - check lighting and that both shoulders are visible.")
        return

    values = [b for b, _ in estimates]
    confidence = statistics.mean(c for _, c in estimates)
    measured, sd = report(rate_bpm, values, "brpm", confidence)
    append_row(csv_path, "breathing", subject, rate_bpm, measured, sd, len(values), confidence)

    path = save_trace(subject, rate_bpm, trace)
    if abs(measured - rate_bpm) > 2.0:
        print(f"\nThat is a long way off. Diagnose it with:\n"
              f"  python tools/validate.py diagnose --trace {path} --rate {rate_bpm:g}")


# ---------------------------------------------------------------------------- pulse


def record_pulse(subject, reference, duration, csv_path):
    from core.heart_rate import HeartRateProcessor

    hr = HeartRateProcessor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available.")
        return

    print(f"Trial: subject={subject} duration={duration}s")
    print("Sit still, face the camera, keep the lighting even.")
    if reference is None:
        print("Count your pulse during the recording - you will be asked for it after.\n")

    start = time.time()
    estimates = []
    total = duration + PULSE_WARMUP_SECONDS

    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= total:
                break
            ok, frame = cap.read()
            if not ok:
                break

            box = hr.add_frame(frame)
            bpm, confidence = hr.estimate_rate()

            warming_up = elapsed < PULSE_WARMUP_SECONDS
            if not warming_up and bpm:
                estimates.append((bpm, confidence))

            preview = frame.copy()
            if box:
                cv2.rectangle(preview, box[:2], box[2:], (0, 200, 120), 2)
            label = f"rPPG: {bpm:.0f} bpm" if bpm else f"buffering {hr.progress():.0%}"
            status_bar(preview, label, total - elapsed, warming_up)

            cv2.imshow("VitalCue pulse - press q to abort", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Aborted.")
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hr.close()

    if not estimates:
        print("No usable estimates - was a face detected throughout?")
        return

    if reference is None:
        raw = input(f"\nBeats counted over {duration}s (blank to discard trial): ").strip()
        if not raw:
            print("Discarded.")
            return
        reference = float(raw) * (60.0 / duration)
        print(f"Reference pulse: {reference:.0f} bpm")

    values = [b for b, _ in estimates]
    confidence = statistics.mean(c for _, c in estimates)
    measured, sd = report(reference, values, "bpm", confidence)
    append_row(csv_path, "pulse", subject, reference, measured, sd, len(values), confidence)


# -------------------------------------------------------------------------- analyze


def plot_trace(trace_path, target_bpm, t, y, fs, out="trace_diagnosis.png"):
    """Four views of a trial: raw signal, band-passed against the expected breath
    timing, spectrum, and what the estimator reported second by second.

    Numbers alone could not settle why a trial failed here - seeing that the raw
    signal had a posture step in the middle of it, and that the breathing excursion
    was smaller than the drift, is what actually identified the problem.
    """
    try:
        import matplotlib  # noqa: PLC0415 - optional, only needed for --plot
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot.")
        return

    sos = butter(3, [RESP_LOW_HZ, RESP_HIGH_HZ], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, y - np.mean(y))

    proc = RespiratoryProcessor()
    times, estimates = [], []
    for ti, yi in zip(t, y):
        proc.add_sample(yi, ti)
        bpm, _ = proc.estimate_rate()
        if bpm:
            times.append(ti)
            estimates.append(bpm)

    fig, ax = plt.subplots(4, 1, figsize=(13, 11))

    ax[0].plot(t, y, lw=0.8, color="#333")
    ax[0].set_title(f"Raw respiratory signal  ({os.path.basename(trace_path)})")
    ax[0].set_ylabel("displacement")

    seg = t <= t[0] + 30
    ax[1].plot(t[seg], filtered[seg], lw=1.0, color="#0a7")
    for k in range(int(30 * target_bpm / 60) + 1):
        ax[1].axvline(t[0] + k * 60.0 / target_bpm, color="#f55", lw=0.6, alpha=0.6)
    ax[1].set_title(f"Band-passed, first 30s. Red lines = where a "
                    f"{target_bpm:g} brpm breath should be")
    ax[1].set_ylabel("excursion")

    n = len(filtered)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs) * 60.0
    power = np.abs(np.fft.rfft(filtered * np.hanning(n))) ** 2
    keep = (freqs >= RESP_LOW_HZ * 60) & (freqs <= 45)
    ax[2].plot(freqs[keep], power[keep] / power[keep].max(), color="#333", lw=1.0)
    ax[2].axvline(target_bpm, color="#f55", lw=1.5, label=f"target {target_bpm:g} brpm")
    ax[2].set_title("Spectrum")
    ax[2].set_xlabel("breaths/min")
    ax[2].legend()

    if estimates:
        ax[3].plot(times, estimates, lw=1.0, color="#06c")
        ax[3].axhline(target_bpm, color="#f55", lw=1.5)
        ax[3].set_ylim(0, max(40, target_bpm * 2))
    ax[3].set_title("What the estimator reported over the trial")
    ax[3].set_xlabel("seconds")
    ax[3].set_ylabel("brpm")

    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print(f"\n  Plot saved to {out}")
    print(f"  band-passed excursion SD {np.std(filtered):.5f} "
          f"(this is the breathing signal's actual size)")
    if estimates:
        print(f"  estimator ranged {min(estimates):.1f} to {max(estimates):.1f} brpm "
              f"during the trial")


def diagnose(trace_path, target_bpm, warmup=RESP_WARMUP_SECONDS, show_plot=False):
    """Show what the estimator was actually looking at during a trial.

    The question a bad trial poses is which of two things happened: the subject
    breathed at the wrong rate, or the subject breathed correctly and the estimator
    picked the wrong spectral peak. The spectrum answers it directly. If there is no
    energy at the target rate, the breathing was not there to find. If there is energy
    at the target but a larger peak elsewhere, the estimator lost a competition.
    """
    with open(trace_path) as f:
        reader = csv.DictReader(f)
        # Older traces used shoulder_y; ROI traces use resp_signal.
        rows = []
        for r in reader:
            value = r.get("resp_signal", r.get("shoulder_y"))
            rows.append((float(r["t"]), float(value)))
    rows = [(t, y) for t, y in rows if t >= warmup]
    if len(rows) < 100:
        print("Not enough trace data to diagnose.")
        return

    t = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    fs = len(t) / (t[-1] - t[0])

    signal = y - np.mean(y)
    windowed = signal * np.hanning(len(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    power = np.abs(np.fft.rfft(windowed)) ** 2

    band = np.flatnonzero((freqs >= RESP_LOW_HZ) & (freqs <= RESP_HIGH_HZ))
    total = float(np.sum(power[band])) + 1e-12

    print(f"\nTrace: {os.path.basename(trace_path)}")
    print(f"  {len(t)} samples over {t[-1] - t[0]:.0f}s at {fs:.1f} fps")
    print(f"  signal range {y.min():.4f} to {y.max():.4f} "
          f"(excursion {y.max() - y.min():.4f})")

    order = band[np.argsort(power[band])[::-1]]
    seen, peaks = set(), []
    for idx in order:
        bpm = freqs[idx] * 60.0
        share = power[idx] / total
        if share < 0.005:  # below this it is spectral floor, not a candidate
            break
        if any(abs(bpm - p) < 2.5 for p in seen):
            continue
        seen.add(bpm)
        peaks.append((bpm, share))
        if len(peaks) == 5:
            break

    print(f"\n  strongest peaks in band (target was {target_bpm:g} brpm):")
    for bpm, share in peaks:
        mark = "  <-- target" if abs(bpm - target_bpm) <= 2.0 else ""
        print(f"    {bpm:6.1f} brpm   {share:5.1%} of band power{mark}")

    at_target = max((s for b, s in peaks if abs(b - target_bpm) <= 2.0), default=0.0)
    winner = peaks[0]
    print()
    if at_target == 0.0:
        print(f"  No peak near {target_bpm:g} brpm. The breathing was not in the signal:")
        print(f"  most likely the pace was not followed, or the shoulders barely moved.")
    elif abs(winner[0] - target_bpm) <= 2.0:
        print("  The target peak won. A bad summary number here means the estimate")
        print("  wandered during the trial rather than locking onto the wrong thing.")
    else:
        print(f"  The target IS present ({at_target:.1%}) but lost to {winner[0]:.1f} brpm "
              f"({winner[1]:.1%}).")
        print("  That is the estimator picking the wrong peak, not a breathing problem.")

    if show_plot:
        plot_trace(trace_path, target_bpm, t, y, fs)


def analyze(csv_path, metric_filter=None):
    if not os.path.exists(csv_path):
        print(f"No results at {csv_path}. Run some trials first.")
        return

    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f)
                if metric_filter is None or r["metric"] == metric_filter]
    if not rows:
        print("No matching trials recorded.")
        return

    for metric in sorted({r["metric"] for r in rows}):
        subset = [r for r in rows if r["metric"] == metric]
        unit = "brpm" if metric == "breathing" else "bpm"
        diffs = [float(r["measured"]) - float(r["reference"]) for r in subset]
        refs = [float(r["reference"]) for r in subset]

        n = len(diffs)
        mae = statistics.mean(abs(d) for d in diffs)
        bias = statistics.mean(diffs)
        sd = statistics.pstdev(diffs) if n > 1 else 0.0
        rmse = math.sqrt(statistics.mean(d * d for d in diffs))
        threshold = 3.0
        within = sum(1 for d in diffs if abs(d) <= threshold) / n

        print(f"\n=== {metric.upper()} ===")
        print(f"Trials {n}   Subjects {len({r['subject'] for r in subset})}")
        print(f"Mean absolute error      {mae:.2f} {unit}")
        print(f"RMSE                     {rmse:.2f} {unit}")
        print(f"Bias (mean difference)   {bias:+.2f} {unit}")
        print(f"95% limits of agreement  {bias - 1.96 * sd:+.2f} to {bias + 1.96 * sd:+.2f} {unit}")
        print(f"Within +/-{threshold:.0f} {unit}          {within:.0%} of trials")

        # Stability within a trial, not just agreement between trials. A trial can hit
        # the target on the median while the live stream the state machine sees wanders.
        sds = [float(r["sd"]) for r in subset if r.get("sd")]
        if sds:
            print(f"Typical within-trial SD    {statistics.mean(sds):.2f} {unit}")

        if len(set(refs)) > 1:
            print("\nPer reference value:")
            for ref in sorted(set(refs)):
                group = [d for d, r in zip(diffs, refs) if r == ref]
                print(f"  {ref:6.0f} {unit}   n={len(group):2d}   "
                      f"MAE {statistics.mean(abs(d) for d in group):.2f}   "
                      f"bias {statistics.mean(group):+.2f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    br = sub.add_parser("breathing", help="paced-breathing trial against an on-screen pacer")
    br.add_argument("--subject", required=True, help="anonymous subject id, e.g. S1")
    br.add_argument("--rate", type=int, required=True,
                    help="target breaths per minute (measurable band is 6-60)")
    br.add_argument("--duration", type=int, default=90, help="recording seconds after warmup")
    br.add_argument("--csv", default=DEFAULT_CSV)

    pu = sub.add_parser("pulse", help="rPPG trial against an external pulse reference")
    pu.add_argument("--subject", required=True, help="anonymous subject id, e.g. S1")
    pu.add_argument("--reference", type=float, default=None,
                    help="known pulse in bpm from a watch or oximeter; prompts if omitted")
    pu.add_argument("--duration", type=int, default=60, help="recording seconds after warmup")
    pu.add_argument("--csv", default=DEFAULT_CSV)

    an = sub.add_parser("analyze", help="summarise all recorded trials")
    an.add_argument("--metric", choices=["breathing", "pulse"], default=None)
    an.add_argument("--csv", default=DEFAULT_CSV)

    dg = sub.add_parser("diagnose", help="inspect the spectrum of one recorded trace")
    dg.add_argument("--trace", required=True, help="path under validation_traces/")
    dg.add_argument("--rate", type=float, required=True, help="the target rate for that trial")
    dg.add_argument("--plot", action="store_true", help="also write trace_diagnosis.png")

    args = parser.parse_args()
    if args.command == "breathing":
        if not 6 <= args.rate <= 60:
            print("Rate must be 6-60 brpm - outside that the band-pass filter removes it.")
            return
        record_breathing(args.subject, args.rate, args.duration, args.csv)
    elif args.command == "pulse":
        record_pulse(args.subject, args.reference, args.duration, args.csv)
    elif args.command == "diagnose":
        diagnose(args.trace, args.rate, show_plot=args.plot)
    else:
        analyze(args.csv, args.metric)


if __name__ == "__main__":
    main()
