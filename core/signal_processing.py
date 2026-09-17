"""
VitalCue - Signal Processing Module
Converts raw shoulder-Y displacement into a respiratory rate (breaths/min).
"""
import time
from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt

# Human respiratory rate range: 9-60 breaths/min -> 0.15-1.0 Hz (Allows detecting hyperventilation/huffing)
RESP_LOW_HZ = 0.15
RESP_HIGH_HZ = 1.0

# Window length matters more than it looks: FFT frequency resolution = 1 / window_seconds.
# A 15s window only resolves to ~4 BPM steps, which is too coarse to separate a normal
# 14 BPM rate from a mildly elevated 18 BPM rate. ~22s gives ~2.7 BPM resolution - a
# meaningful accuracy gain for a small added latency cost.
WINDOW_SECONDS = 22
MIN_SAMPLES_FOR_ESTIMATE = 8 * 30  # ~8s of frames at ~30fps, floor before first reading


class RespiratoryProcessor:
    """Buffers shoulder-Y samples and estimates breathing rate via band-pass + FFT."""

    def __init__(self, window_seconds: float = WINDOW_SECONDS, fps_hint: float = 30.0):
        self.window_seconds = window_seconds
        self.fps_hint = fps_hint
        maxlen = int(window_seconds * fps_hint * 1.5)  # headroom for jittery fps
        self.samples = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)

    def add_sample(self, shoulder_y, t: float = None):
        if shoulder_y is None:
            return
        self.samples.append(shoulder_y)
        self.timestamps.append(t if t is not None else time.time())

    def _smoothed_signal(self, raw: np.ndarray) -> np.ndarray:
        """Light moving-average smoothing to reduce per-frame pose-detection jitter
        before filtering - raw landmark noise otherwise pollutes the FFT peak."""
        kernel_size = 5
        if len(raw) < kernel_size:
            return raw
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(raw, kernel, mode="same")

    def estimate_rate(self):
        """
        Returns (breaths_per_minute, confidence) or (None, 0.0) if not enough/usable data.
        confidence is a rough 0-1 score based on how dominant the peak frequency is
        relative to total power in the respiratory band.
        """
        if len(self.samples) < MIN_SAMPLES_FOR_ESTIMATE:
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
            b, a = butter(N=3, Wn=[RESP_LOW_HZ, RESP_HIGH_HZ], btype="band", fs=fs)
            filtered = filtfilt(b, a, signal)
        except ValueError:
            return None, 0.0

        n = len(filtered)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        power = np.abs(np.fft.rfft(filtered)) ** 2

        band_mask = (freqs >= RESP_LOW_HZ) & (freqs <= RESP_HIGH_HZ)
        if not np.any(band_mask):
            return None, 0.0

        band_freqs = freqs[band_mask]
        band_power = power[band_mask]

        peak_idx = int(np.argmax(band_power))
        peak_freq = band_freqs[peak_idx]
        peak_power = band_power[peak_idx]
        total_power = float(np.sum(band_power)) + 1e-9
        confidence = float(peak_power / total_power)

        bpm = float(peak_freq * 60.0)
        return bpm, confidence

    def is_ready(self) -> bool:
        return len(self.samples) >= MIN_SAMPLES_FOR_ESTIMATE

    def get_waveform(self):
        """Returns the most recent smoothed samples for live graphing.
        Inverts Y so taking a breath (shoulders go up, Y goes down) shows as an upward spike."""
        if len(self.samples) < 5:
            return []
        # Return the last ~5 seconds of data (150 frames)
        raw = np.array(self.samples)[-150:].astype(float)
        smoothed = self._smoothed_signal(raw)
        # Detrend locally so posture shifts don't blow up the scale
        local_mean = np.mean(smoothed)
        detrended = smoothed - local_mean
        return (-detrended).tolist()
