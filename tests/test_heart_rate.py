"""
Heart rate signal-path tests.

These cover the preprocessing and frequency estimation around PhysNet, not the network
itself — a pretrained checkpoint's accuracy is a question for the validation harness
against a real pulse, not for a unit test. What is worth pinning down here is that the
maths surrounding the model does not silently corrupt its input or misread its output.
"""
import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.heart_rate import (  # noqa: E402
    CHUNK_FRAMES,
    HR_HIGH_HZ,
    HR_LOW_HZ,
    TARGET_FPS,
    WINDOW_SECONDS,
    HeartRateProcessor,
)


def synthetic_bvp(hr_bpm, seconds, fs):
    t = np.arange(0, seconds, 1.0 / fs)
    return np.sin(2 * np.pi * (hr_bpm / 60.0) * t)


class TestRateFromWaveform(unittest.TestCase):
    def assert_reads(self, hr_bpm, tolerance=3.0):
        bvp = synthetic_bvp(hr_bpm, WINDOW_SECONDS, TARGET_FPS)
        bpm, confidence = HeartRateProcessor._bpm_from_bvp(bvp, TARGET_FPS)
        self.assertIsNotNone(bpm)
        self.assertAlmostEqual(
            bpm, hr_bpm, delta=tolerance,
            msg=f"read {bpm:.1f} bpm from a {hr_bpm} bpm waveform",
        )
        self.assertGreater(confidence, 0.0)

    def test_resting_pulse(self):
        self.assert_reads(60)

    def test_typical_pulse(self):
        self.assert_reads(75)

    def test_elevated_pulse(self):
        self.assert_reads(110)

    def test_sub_bin_resolution_is_recovered(self):
        """A 4.27s window is ~14 bpm per FFT bin. Without parabolic interpolation
        these three rates would all collapse onto the same bin."""
        readings = []
        for hr in (68, 72, 76):
            bvp = synthetic_bvp(hr, WINDOW_SECONDS, TARGET_FPS)
            readings.append(HeartRateProcessor._bpm_from_bvp(bvp, TARGET_FPS)[0])
        self.assertTrue(
            readings[0] < readings[1] < readings[2],
            f"interpolation did not separate nearby rates: {readings}",
        )

    def test_out_of_band_signal_is_not_reported_as_pulse(self):
        """A slow postural sway must not be read as a very low heart rate."""
        bvp = synthetic_bvp(20, WINDOW_SECONDS, TARGET_FPS)  # 0.33 Hz, below the band
        bpm, _ = HeartRateProcessor._bpm_from_bvp(bvp, TARGET_FPS)
        self.assertGreaterEqual(bpm, HR_LOW_HZ * 60 - 1)


class TestDiffNormalize(unittest.TestCase):
    @staticmethod
    def pulsing_frames(hr_bpm=72, n=CHUNK_FRAMES, brightness=1.0):
        t = np.arange(n) / TARGET_FPS
        pulse = 0.02 * np.sin(2 * np.pi * (hr_bpm / 60.0) * t)
        base = np.ones((n, 8, 8, 3), dtype=np.float32) * 0.5 * brightness
        return base * (1.0 + pulse[:, None, None, None])

    def test_output_shape_matches_input(self):
        frames = self.pulsing_frames()
        out = HeartRateProcessor._diff_normalize(frames)
        self.assertEqual(out.shape, frames.shape)

    def test_no_nan_or_inf_from_dark_frames(self):
        frames = np.zeros((16, 8, 8, 3), dtype=np.float32)
        out = HeartRateProcessor._diff_normalize(frames)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_illumination_scale_is_cancelled(self):
        """Dividing the difference by the sum is what makes the network robust to
        how brightly the room is lit. A dimmer feed must yield the same signal."""
        bright = HeartRateProcessor._diff_normalize(self.pulsing_frames(brightness=1.0))
        dim = HeartRateProcessor._diff_normalize(self.pulsing_frames(brightness=0.4))
        np.testing.assert_allclose(bright, dim, atol=1e-4)


class TestResampling(unittest.TestCase):
    class FakeProcessor:
        """Exercises the resampling maths without loading the face detector."""

        def __init__(self, crops, timestamps):
            self.crops = crops
            self.timestamps = timestamps
            self.chunk_frames = CHUNK_FRAMES

        _select_frames = HeartRateProcessor._select_frames
        _resampled_chunk = HeartRateProcessor._resampled_chunk

    def build(self, real_fps):
        n = int(WINDOW_SECONDS * real_fps)
        crops = [np.full((4, 4, 3), i, dtype=np.float32) for i in range(n)]
        timestamps = [i / real_fps for i in range(n)]
        return self.FakeProcessor(crops, timestamps)

    def test_low_frame_rate_is_upsampled_to_the_model_chunk_length(self):
        chunk = self.build(11.0)._resampled_chunk()
        self.assertEqual(len(chunk), CHUNK_FRAMES)

    def test_high_frame_rate_is_downsampled_to_the_model_chunk_length(self):
        chunk = self.build(60.0)._resampled_chunk()
        self.assertEqual(len(chunk), CHUNK_FRAMES)

    def test_temporal_order_is_preserved(self):
        chunk = self.build(11.0)._resampled_chunk()
        values = [c[0, 0, 0] for c in chunk]
        self.assertEqual(values, sorted(values), "resampling scrambled frame order")

    def test_endpoints_are_retained(self):
        proc = self.build(11.0)
        chunk = proc._resampled_chunk()
        self.assertEqual(chunk[0][0, 0, 0], proc.crops[0][0, 0, 0])
        self.assertEqual(chunk[-1][0, 0, 0], proc.crops[-1][0, 0, 0])


class TestNonBlockingInference(unittest.TestCase):
    """The whole point of the worker thread is that the frame loop never waits on it.

    Driven through a stand-in rather than a real processor so the test needs neither
    the face detector model nor torch, but the method under test is the real one.
    """

    class FakeProcessor:
        def __init__(self, infer_seconds=0.4):
            self.chunk_frames = CHUNK_FRAMES
            self.infer_interval = 0.0
            self.infer_seconds = infer_seconds
            self.crops = [np.zeros((2, 2, 3), dtype=np.float32)] * CHUNK_FRAMES
            self.timestamps = list(np.linspace(0, WINDOW_SECONDS, CHUNK_FRAMES))
            self._lock = threading.Lock()
            self._worker = None
            self._closed = False
            self._cached = (None, 0.0)
            self._last_inference = 0.0
            self.infer_calls = 0

        def _is_ready_locked(self):
            return True

        def _infer(self, selected, duration):
            time.sleep(self.infer_seconds)
            with self._lock:
                self.infer_calls += 1
                self._cached = (72.0, 0.5)

        _select_frames = HeartRateProcessor._select_frames
        estimate_rate = HeartRateProcessor.estimate_rate

    def test_caller_is_never_blocked_by_inference(self):
        proc = self.FakeProcessor(infer_seconds=0.4)
        worst = 0.0
        start = time.perf_counter()
        while time.perf_counter() - start < 0.5:
            call = time.perf_counter()
            proc.estimate_rate()
            worst = max(worst, time.perf_counter() - call)
        proc._worker.join()
        self.assertLess(worst, 0.05,
                        f"a call blocked for {worst * 1000:.0f} ms while inference ran")

    def test_only_one_inference_runs_at_a_time(self):
        proc = self.FakeProcessor(infer_seconds=0.3)
        for _ in range(200):
            proc.estimate_rate()
        proc._worker.join()
        self.assertEqual(proc.infer_calls, 1)

    def test_previous_estimate_is_served_while_recomputing(self):
        proc = self.FakeProcessor(infer_seconds=0.2)
        proc.estimate_rate()
        proc._worker.join()
        self.assertEqual(proc.estimate_rate()[0], 72.0)

    def test_no_inference_starts_after_close(self):
        proc = self.FakeProcessor()
        proc._closed = True
        self.assertEqual(proc.estimate_rate(), (None, 0.0))
        self.assertIsNone(proc._worker)


class TestBandConfiguration(unittest.TestCase):
    def test_band_spans_a_plausible_human_pulse_range(self):
        self.assertLessEqual(HR_LOW_HZ * 60, 45)
        self.assertGreaterEqual(HR_HIGH_HZ * 60, 170)

    def test_window_matches_the_models_training_configuration(self):
        self.assertAlmostEqual(WINDOW_SECONDS, CHUNK_FRAMES / TARGET_FPS, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
