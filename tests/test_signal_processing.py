"""
Signal chain tests against synthetic breathing.

A clean sine wave at a known frequency is the easiest possible input, so these are
a floor, not a validation — real accuracy comes from tools/validate.py.
Their job is to catch a broken filter or an off-by-one in the FFT bin mapping.
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.signal_processing import (  # noqa: E402
    RESP_HIGH_HZ,
    RESP_LOW_HZ,
    RespiratoryProcessor,
)

FPS = 30.0


def synthetic_breathing(processor, rate_bpm, seconds, amplitude=0.01):
    """Feed a sine wave at the given breathing rate on a fixed clock."""
    freq_hz = rate_bpm / 60.0
    for i in range(int(seconds * FPS)):
        t = i / FPS
        processor.add_sample(0.5 + amplitude * math.sin(2 * math.pi * freq_hz * t), t)
    return processor


class TestRateEstimation(unittest.TestCase):
    def assert_estimates(self, rate_bpm, tolerance=2.0, seconds=40):
        proc = synthetic_breathing(RespiratoryProcessor(), rate_bpm, seconds)
        bpm, confidence = proc.estimate_rate()
        self.assertIsNotNone(bpm, f"no estimate produced at {rate_bpm} brpm")
        self.assertAlmostEqual(
            bpm, rate_bpm, delta=tolerance,
            msg=f"estimated {bpm:.1f} for a {rate_bpm} brpm signal",
        )
        self.assertGreater(confidence, 0.05)

    def test_resting_rate(self):
        self.assert_estimates(12)

    def test_slightly_elevated_rate(self):
        self.assert_estimates(18)

    def test_fast_rate(self):
        self.assert_estimates(24)

    def test_paced_breathing_target_is_measurable(self):
        """The intervention coaches toward ~6 brpm. With the original 0.15 Hz floor
        (9 brpm) this rate was filtered out entirely, so the system could never
        confirm the person actually complied with its own instruction."""
        self.assert_estimates(6, tolerance=1.0, seconds=60)


class TestSmoothingEdges(unittest.TestCase):
    """Regression guard for the zero-padding bug.

    Shoulder-Y sits near 0.5 while the breathing excursion is about 0.01. Smoothing
    with numpy's mode="same" pads with implicit zeros, dragging the end samples from
    0.5 toward 0 — a step forty times the size of the signal. That step is pure
    low-frequency energy, and it pinned every estimate to the lowest in-band bin
    regardless of how the subject actually breathed.
    """

    def test_smoothing_does_not_amplify_signal_range(self):
        proc = RespiratoryProcessor()
        raw = 0.5 + 0.01 * np.sin(np.linspace(0, 8 * np.pi, 600))
        smoothed = proc._smoothed_signal(raw)
        self.assertLessEqual(
            np.ptp(smoothed), np.ptp(raw) * 1.05,
            "smoothing widened the signal range — edge padding has regressed",
        )

    def test_smoothing_preserves_the_dc_offset_at_the_edges(self):
        proc = RespiratoryProcessor()
        raw = np.full(600, 0.5)
        smoothed = proc._smoothed_signal(raw)
        self.assertAlmostEqual(smoothed[0], 0.5, places=6)
        self.assertAlmostEqual(smoothed[-1], 0.5, places=6)

    def test_distinct_rates_produce_distinct_estimates(self):
        rates = [10, 16, 22]
        estimates = []
        for rate in rates:
            proc = synthetic_breathing(RespiratoryProcessor(), rate, seconds=45)
            estimates.append(proc.estimate_rate()[0])
        self.assertEqual(len(set(round(e) for e in estimates)), len(rates),
                         f"different breathing rates collapsed to {estimates}")


class TestEffortMeasurement(unittest.TestCase):
    """Effort exists because rate cannot see breathing hard at an unchanged speed."""

    @staticmethod
    def measure(rate_bpm, amplitude, seconds=30, fps=15.0):
        proc = RespiratoryProcessor()
        phase = 0.0
        for i in range(int(seconds * fps)):
            phase += 2 * np.pi * (rate_bpm / 60.0) / fps
            proc.add_sample(0.5 + amplitude * np.sin(phase), i / fps)
        bpm, _ = proc.estimate_rate()
        return bpm, proc.estimate_effort()

    def test_effort_is_proportional_to_excursion(self):
        ratios = [self.measure(14, amp)[1] / amp for amp in (0.005, 0.01, 0.02, 0.04)]
        for ratio in ratios[1:]:
            self.assertAlmostEqual(ratio, ratios[0], delta=0.05 * ratios[0])

    def test_effort_is_independent_of_breathing_rate(self):
        """Otherwise a person simply breathing faster would read as breathing harder."""
        efforts = [self.measure(rate, 0.02)[1] for rate in (10, 14, 20, 28)]
        self.assertLess((max(efforts) - min(efforts)) / max(efforts), 0.10)

    def test_rate_is_unchanged_when_only_depth_changes(self):
        """The regression this whole signal exists to cover."""
        shallow, _ = self.measure(14, 0.01)
        deep, _ = self.measure(14, 0.05)
        self.assertAlmostEqual(shallow, deep, delta=0.5)

    def test_deeper_breathing_does_raise_effort(self):
        _, shallow = self.measure(14, 0.01)
        _, deep = self.measure(14, 0.05)
        self.assertGreater(deep, shallow * 3)

    def test_effort_is_none_before_any_estimate(self):
        self.assertIsNone(RespiratoryProcessor().estimate_effort())


class TestGuards(unittest.TestCase):
    def test_band_covers_the_paced_breathing_target(self):
        self.assertLessEqual(RESP_LOW_HZ, 6.0 / 60.0)
        self.assertGreaterEqual(RESP_HIGH_HZ, 30.0 / 60.0)

    def test_no_estimate_before_minimum_samples(self):
        proc = synthetic_breathing(RespiratoryProcessor(), 12, seconds=3)
        bpm, confidence = proc.estimate_rate()
        self.assertIsNone(bpm)
        self.assertEqual(confidence, 0.0)

    def test_none_samples_are_dropped(self):
        proc = RespiratoryProcessor()
        for i in range(10):
            proc.add_sample(None, i / FPS)
        self.assertFalse(proc.is_ready())

    def test_waveform_preserves_inhale_as_upward(self):
        """ROI flow already signs inhale positive; the display must not flip it back."""
        proc = RespiratoryProcessor()
        for i in range(60):
            proc.add_sample(0.0 + i * 0.001, i / FPS)
        waveform = proc.get_waveform()
        self.assertGreater(waveform[-1], waveform[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
