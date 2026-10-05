"""
VitalCue - Signal Processing Module
Converts a raw respiratory displacement signal into a rate (breaths/min) and an
effort measure (how far the chest travels per breath).

The upstream signal used to be the Y coordinate of two shoulder landmarks. It is now
the integrated vertical optical flow inside a chest ROI - same units in spirit
(a displacement that rises on inhale), but far more sensitive to shallow breathing.
This module does not care which; it only needs an oscillatory displacement.

Rate alone misses a whole class of behaviour. Breathing hard without breathing fast
leaves the rate estimate flat - measured, five times the excursion at the same speed
still reads the same BPM - so anything driven off rate alone simply cannot see it.
Effort is the amplitude of that same band-passed signal, which costs almost nothing
extra because the filtering has already been done to get the rate.
"""
import time
from collections import deque

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Band must cover paced-breathing rates, not just resting ones. The intervention guides
# people toward roughly 6 breaths/min (0.1 Hz), so the floor has to sit *below* that -
# at exactly 0.1 Hz the target rate lands on the filter's own roll-off and reads about
# 1.3 BPM high. 0.08 Hz puts it inside the passband. Going lower buys nothing and starts
# admitting slow postural drift.
RESP_LOW_HZ = 0.08
RESP_HIGH_HZ = 1.0

# Window length matters more than it looks: FFT frequency resolution = 1 / window_seconds.
# A 15s window only resolves to ~4 BPM steps, which is too coarse to separate a normal
# 14 BPM rate from a mildly elevated 18 BPM rate. ~22s gives ~2.7 BPM resolution - a
# meaningful accuracy gain for a small added latency cost.
WINDOW_SECONDS = 22

# Both of these are in seconds, and the buffer is trimmed by timestamp rather than by
# sample count. Sizing the deque by frames assumes a frame rate: MediaPipe pose on a
# laptop webcam runs nearer 17fps than 30, so a count-based buffer of 22s * 30fps held
# almost a full minute of data - triple the intended window, with the latency to match.
MIN_SECONDS_FOR_ESTIMATE = 8.0
MIN_SAMPLES_FOR_ESTIMATE = 60   # guard against a very low frame rate faking a full window


class RespiratoryProcessor:
    """Buffers shoulder-Y samples and estimates breathing rate via band-pass + FFT."""

    def __init__(self, window_seconds: float = WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self.samples = deque()
        self.timestamps = deque()
        self._last_effort = None

    def add_sample(self, displacement, t: float = None):
        if displacement is None:
            return
        self.samples.append(displacement)
        self.timestamps.append(t if t is not None else time.time())
        # Trim by age so the window is a real duration at whatever fps we get.
        while len(self.timestamps) > 1 and self.timestamps[-1] - self.timestamps[0] > self.window_seconds:
            self.timestamps.popleft()
            self.samples.popleft()

    def _smoothed_signal(self, raw: np.ndarray) -> np.ndarray:
        """Light moving-average smoothing to reduce per-frame pose-detection jitter
        before filtering - raw landmark noise otherwise pollutes the FFT peak.

        Edge-padded rather than zero-padded. Shoulder-Y sits around 0.5 while the
        breathing excursion is roughly 0.01, so convolving with implicit zeros
        (numpy's mode="same") drags the first and last samples from 0.5 toward 0
        and injects a step forty times larger than the signal. That step is pure
        low-frequency energy and it dominates the FFT, which made every estimate
        collapse onto the lowest in-band bin regardless of how the person breathed.
        """
        kernel_size = 5
        if len(raw) < kernel_size:
            return raw
        pad = kernel_size // 2
        kernel = np.ones(kernel_size) / kernel_size
        padded = np.pad(raw, pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _interpolated_peak(freqs, power, idx: int) -> float:
        """Sub-bin peak location by parabolic fit over the log spectrum.

        FFT resolution is 1/window, about 2.7 BPM for a 22s window, which is too
        coarse to separate a normal rate from a mildly elevated one. Fitting a
        parabola through the peak bin and its neighbours recovers most of that.
        """
        if idx <= 0 or idx >= len(power) - 1:
            return float(freqs[idx])
        a, b, c = np.log(power[idx - 1: idx + 2] + 1e-20)
        denom = a - 2.0 * b + c
        if denom == 0:
            return float(freqs[idx])
        delta = 0.5 * (a - c) / denom
        if abs(delta) > 0.5:
            return float(freqs[idx])
        return float(freqs[idx] + delta * (freqs[1] - freqs[0]))

    def estimate_rate(self):
        """
        Returns (breaths_per_minute, confidence) or (None, 0.0) if not enough/usable data.
        confidence is a rough 0-1 score based on how dominant the peak frequency is
        relative to total power in the respiratory band.
        """
        if not self.is_ready():
            return None, 0.0

        raw = np.array(self.samples, dtype=float)
        ts = np.array(self.timestamps, dtype=float)
        duration = ts[-1] - ts[0]
        if duration <= 0:
            return None, 0.0

        fs = len(raw) / duration  # effective sampling rate
        if fs <= 2 * RESP_HIGH_HZ:
            return None, 0.0  # not enough samples/sec to resolve the band we care about

        signal = self._smoothed_signal(raw)
        signal = signal - np.mean(signal)

        try:
            # Second-order sections rather than transfer-function coefficients:
            # the cutoffs are a tiny fraction of Nyquist here, which is where the
            # b/a form loses numerical precision.
            sos = butter(N=3, Wn=[RESP_LOW_HZ, RESP_HIGH_HZ], btype="band", fs=fs, output="sos")
            filtered = sosfiltfilt(sos, signal)
        except ValueError:
            return None, 0.0

        n = len(filtered)
        # Taper the ends so the finite window doesn't smear the peak across bins.
        windowed = filtered * np.hanning(n)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        power = np.abs(np.fft.rfft(windowed)) ** 2

        band_idx = np.flatnonzero((freqs >= RESP_LOW_HZ) & (freqs <= RESP_HIGH_HZ))
        if band_idx.size == 0:
            return None, 0.0

        peak_idx = int(band_idx[np.argmax(power[band_idx])])
        peak_freq = self._interpolated_peak(freqs, power, peak_idx)
        total_power = float(np.sum(power[band_idx])) + 1e-9
        confidence = float(power[peak_idx] / total_power)

        self._last_effort = self._peak_amplitude(power, peak_idx, n)

        bpm = float(peak_freq * 60.0)
        return bpm, confidence

    @staticmethod
    def _peak_amplitude(power, peak_idx: int, n: int) -> float:
        """Amplitude of the breathing component, proportional to shoulder excursion.

        Taken from the bins around the spectral peak rather than as the RMS of the
        whole band-passed signal, because pose-landmark jitter is broadband and would
        otherwise read as breathing effort. The Hann window spreads a pure tone over
        about three bins, hence the neighbourhood; the coherent-gain divisor undoes
        the window's attenuation.

        Summing three bins overshoots a pure tone by a constant ~1.21x, and the
        absolute scale depends on how far the person sits from the camera anyway, so
        this number is only ever meaningful relative to that person's own baseline.
        Measured linear in excursion to within 1% and flat across 10-28 brpm to
        within 4%, which is what the baseline comparison relies on.
        """
        lo = max(0, peak_idx - 1)
        magnitude = float(np.sqrt(np.sum(power[lo: peak_idx + 2])))
        coherent_gain = float(np.sum(np.hanning(n))) / n
        return 2.0 * magnitude / (n * coherent_gain)

    def estimate_effort(self):
        """Breathing excursion from the most recent estimate_rate() call, or None.

        Deliberately a cached read rather than its own computation: it is derived from
        the same FFT, and recomputing it would mean band-pass filtering twice per frame.
        """
        return self._last_effort

    def is_ready(self) -> bool:
        """Enough elapsed time, and enough samples within it, to attempt an estimate."""
        if len(self.samples) < MIN_SAMPLES_FOR_ESTIMATE:
            return False
        return self.timestamps[-1] - self.timestamps[0] >= MIN_SECONDS_FOR_ESTIMATE

    def buffered_seconds(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def effective_fps(self) -> float:
        duration = self.buffered_seconds()
        return len(self.samples) / duration if duration > 0 else 0.0

    def get_waveform(self, seconds: float = 8.0):
        """Most recent smoothed samples for live graphing.

        The ROI flow signal is already signed so inhale is positive; the old landmark
        path needed an invert because image Y grows downward. No invert here.
        """
        if len(self.samples) < 5:
            return []
        count = max(5, int(seconds * self.effective_fps())) if self.effective_fps() else len(self.samples)
        raw = np.array(self.samples)[-count:].astype(float)
        smoothed = self._smoothed_signal(raw)
        # Detrend locally so posture shifts don't blow up the scale
        local_mean = np.mean(smoothed)
        return (smoothed - local_mean).tolist()
