"""
Chest-ROI respiratory signal tests.

These cover the geometry and the optical-flow sensitivity claim without needing a
webcam or the pose model. The integration that motivated the switch - a shallow
breathing peak losing to postural sway - is reproduced as synthetic frames.
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vision import (  # noqa: E402
    ROI_SIZE,
    chest_roi_from_landmarks,
    mean_vertical_flow,
)


class FakeLandmark:
    def __init__(self, x, y, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def fake_pose(left_s=(0.35, 0.40), right_s=(0.65, 0.40),
              left_h=(0.40, 0.75), right_h=(0.60, 0.75), hip_vis=1.0):
    """Minimal landmark list with the four torso points the ROI needs."""
    lm = [FakeLandmark(0.5, 0.5, 0.0) for _ in range(33)]
    lm[11] = FakeLandmark(*left_s)
    lm[12] = FakeLandmark(*right_s)
    lm[23] = FakeLandmark(*left_h, visibility=hip_vis)
    lm[24] = FakeLandmark(*right_h, visibility=hip_vis)
    return lm


class TestChestRoi(unittest.TestCase):
    def test_roi_sits_between_the_shoulders_and_above_the_hips(self):
        box = chest_roi_from_landmarks(fake_pose(), 640, 480)
        self.assertIsNotNone(box)
        x0, y0, x1, y1 = box
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)
        # Shoulders at y=0.40 * 480 = 192; hips at 0.75 * 480 = 360.
        self.assertGreater(y0, 100)
        self.assertLess(y1, 360)

    def test_roi_is_none_when_shoulders_are_invisible(self):
        lm = fake_pose()
        lm[11].visibility = 0.1
        self.assertIsNone(chest_roi_from_landmarks(lm, 640, 480))

    def test_falls_back_without_hips(self):
        box = chest_roi_from_landmarks(fake_pose(hip_vis=0.0), 640, 480)
        self.assertIsNotNone(box)


class TestOpticalFlowSensitivity(unittest.TestCase):
    """The claim: averaging flow over a patch recovers motion a single landmark misses."""

    @staticmethod
    def textured_roi(size=ROI_SIZE):
        """High-contrast texture Farneback can lock onto (a smooth sine band is not)."""
        w, h = size
        rng = np.random.default_rng(0)
        noise = rng.integers(40, 220, (h, w), dtype=np.uint8)
        # Soft vertical gradient on top so a pure vertical shift changes the patch.
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        graded = (noise.astype(np.float32) * (0.55 + 0.45 * yy)).astype(np.uint8)
        return graded

    @staticmethod
    def shifted(base, dy_px):
        """Sub-pixel vertical shift via affine warp — the real motion Farneback sees."""
        h, w = base.shape
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, dy_px]], dtype=np.float32)
        return cv2.warpAffine(base, matrix, (w, h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def test_upward_motion_reads_positive(self):
        base = self.textured_roi()
        # Image Y grows downward, so a rising chest is a negative pixel shift.
        self.assertGreater(mean_vertical_flow(base, self.shifted(base, -2.0)), 0.0)

    def test_downward_motion_reads_negative(self):
        base = self.textured_roi()
        self.assertLess(mean_vertical_flow(base, self.shifted(base, +2.0)), 0.0)

    def test_roi_flow_recovers_sub_pixel_breathing_that_landmarks_lose(self):
        """Reproduce the validation failure mode synthetically.

        A landmark quantised to whole pixels cannot see a 0.4px chest rise. Flow over
        a 96x72 patch can, because it averages thousands of estimates.
        """
        fps, seconds, rate = 20.0, 20.0, 20.0
        base = self.textured_roi()
        landmark_quantised = []
        flow_integrated = 0.0
        flow_series = []
        prev = base

        for i in range(int(seconds * fps)):
            t = i / fps
            # 0.4px peak-to-peak - below landmark quantisation, above flow noise floor.
            true_shift = 0.2 * np.sin(2 * np.pi * (rate / 60.0) * t)
            landmark_quantised.append(0.40 + round(true_shift) / 480.0)
            curr = self.shifted(base, true_shift)
            flow_integrated = 0.995 * flow_integrated + mean_vertical_flow(prev, curr)
            flow_series.append(flow_integrated)
            prev = curr

        landmark_amp = float(np.ptp(landmark_quantised))
        flow_amp = float(np.ptp(flow_series))
        self.assertEqual(landmark_amp, 0.0,
                         "the landmark path should be blind to sub-pixel motion")
        self.assertGreater(flow_amp, 0.05,
                           f"ROI flow amp {flow_amp:.3f} should clearly exceed noise")

        n = len(flow_series)
        signal = np.array(flow_series) - np.mean(flow_series)
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        power = np.abs(np.fft.rfft(signal * np.hanning(n))) ** 2
        peak = int(np.argmax(power[1:]) + 1)
        # Same sub-bin fit the production estimator uses - a 20s window is still
        # only 3 bpm per bin, coarse enough to miss 20 without it.
        if 0 < peak < len(power) - 1:
            a, b, c = np.log(power[peak - 1: peak + 2] + 1e-20)
            denom = a - 2.0 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
            delta = delta if abs(delta) <= 0.5 else 0.0
        else:
            delta = 0.0
        peak_hz = freqs[peak] + delta * (freqs[1] - freqs[0])
        self.assertAlmostEqual(peak_hz * 60.0, rate, delta=2.0)


class TestFarnebackAvailable(unittest.TestCase):
    def test_opencv_farneback_runs_on_roi_sized_patches(self):
        a = np.random.randint(0, 255, (ROI_SIZE[1], ROI_SIZE[0]), dtype=np.uint8)
        b = np.random.randint(0, 255, (ROI_SIZE[1], ROI_SIZE[0]), dtype=np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, 0.5, 2, 15, 2, 5, 1.1, 0)
        self.assertEqual(flow.shape, (ROI_SIZE[1], ROI_SIZE[0], 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
