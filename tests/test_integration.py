"""
End-to-end tests: synthetic shoulder motion -> signal processor -> state machine.

The unit tests either feed the engine effort numbers directly or check the processor
in isolation, so neither of them can catch the two halves disagreeing. These drive the
real pipeline from a simulated shoulder trace and assert on the state that comes out,
which is the claim the product actually makes.

Simulated time throughout: a demo-mode calibration alone is 25 seconds of wall clock.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import engine as engine_module  # noqa: E402
from core.engine import BaselineEngine, VitalState  # noqa: E402
from core.signal_processing import RespiratoryProcessor  # noqa: E402

FPS = 15.0
CALM_RATE = 14.0
CALM_AMPLITUDE = 0.01


class Session:
    """Drives both components off one simulated clock and one synthetic trace."""

    def __init__(self, demo_mode=True):
        self.clock = 1000.0
        self.phase = 0.0
        self.resp = RespiratoryProcessor()
        self.engine = BaselineEngine(demo_mode=demo_mode)
        self.state = VitalState.CALIBRATING

    def breathe(self, seconds, rate_bpm=CALM_RATE, amplitude=CALM_AMPLITUDE):
        for _ in range(int(seconds * FPS)):
            self.phase += 2 * np.pi * (rate_bpm / 60.0) / FPS
            self.clock += 1.0 / FPS
            self.resp.add_sample(0.5 + amplitude * np.sin(self.phase), self.clock)
            bpm, confidence = self.resp.estimate_rate()
            if bpm:
                self.state = self.engine.update(
                    bpm, confidence, self.resp.estimate_effort())
        return self.state

    def seconds_until(self, predicate, limit, **breathe_kwargs):
        """Returns seconds of simulated breathing until predicate(state) holds."""
        for elapsed in range(int(limit)):
            self.breathe(1.0, **breathe_kwargs)
            if predicate(self.state):
                return elapsed + 1
        return None


class IntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self._real_time = engine_module.time.time
        self.session = Session()
        engine_module.time.time = lambda: self.session.clock

    def tearDown(self):
        engine_module.time.time = self._real_time

    def calibrated(self):
        self.session.breathe(40)
        self.assertIsNotNone(self.session.engine.baseline_mean,
                             "calibration did not complete")
        self.assertIsNotNone(self.session.engine.baseline_effort_mean,
                             "effort baseline was not established")
        return self.session


class TestCalmBreathing(IntegrationTestCase):
    def test_steady_breathing_stays_normal(self):
        session = self.calibrated()
        self.assertEqual(session.breathe(60), VitalState.NORMAL)

    def test_baseline_lands_near_the_true_rate(self):
        session = self.calibrated()
        self.assertAlmostEqual(session.engine.baseline_mean, CALM_RATE, delta=2.0)


class TestHardBreathing(IntegrationTestCase):
    """The gap this feature was added to close."""

    def test_breathing_harder_at_the_same_rate_reaches_stress(self):
        session = self.calibrated()
        state = session.breathe(45, rate_bpm=CALM_RATE, amplitude=CALM_AMPLITUDE * 4)
        self.assertEqual(state, VitalState.STRESS)

    def test_the_rate_really_does_stay_flat_while_that_happens(self):
        """Proves the trigger came from effort and not from a rate change."""
        session = self.calibrated()
        session.breathe(45, rate_bpm=CALM_RATE, amplitude=CALM_AMPLITUDE * 4)
        self.assertLess(session.engine.last_z, engine_module.STRESS_Z_THRESHOLD)
        self.assertGreaterEqual(session.engine.last_effort_z,
                                engine_module.EFFORT_Z_THRESHOLD)

    def test_softening_the_breathing_resolves_the_episode(self):
        session = self.calibrated()
        session.breathe(45, amplitude=CALM_AMPLITUDE * 4)
        state = session.breathe(60)
        self.assertEqual(state, VitalState.NORMAL)
        self.assertTrue(session.engine.recovery_events)
        self.assertTrue(session.engine.recovery_events[-1]["resolved"])


class TestPacedBreathingIsNotDistress(IntegrationTestCase):
    """Regression: deep *slow* breathing is what the intervention coaches, and it
    produces a very large excursion. An effort trigger with no rate guard fired on it
    within 19 seconds, so following the guidance re-triggered the alert that prompted
    it and the app fought its own intervention."""

    PACED_RATE = 7.0
    DEEP = CALM_AMPLITUDE * 3

    def test_slow_deep_breathing_never_triggers_stress(self):
        session = self.calibrated()
        state = session.breathe(90, rate_bpm=self.PACED_RATE, amplitude=self.DEEP)
        self.assertEqual(state, VitalState.NORMAL)

    def test_effort_really_is_high_during_that(self):
        """Guards the guard: if effort stopped rising here the test above would pass
        for the wrong reason."""
        session = self.calibrated()
        session.breathe(60, rate_bpm=self.PACED_RATE, amplitude=self.DEEP)
        self.assertGreater(session.engine.last_effort_z,
                           engine_module.EFFORT_Z_THRESHOLD)
        self.assertLess(session.engine.last_z, 0)

    def test_following_the_guidance_lets_you_leave_stress(self):
        """The same bug on the exit path would hold someone in STRESS for responding."""
        session = self.calibrated()
        session.breathe(45, amplitude=CALM_AMPLITUDE * 4)
        self.assertEqual(session.state, VitalState.STRESS)
        state = session.breathe(60, rate_bpm=self.PACED_RATE, amplitude=self.DEEP)
        self.assertEqual(state, VitalState.NORMAL)


class TestFastBreathing(IntegrationTestCase):
    def test_rapid_shallow_breathing_reaches_stress(self):
        session = self.calibrated()
        state = session.breathe(45, rate_bpm=26.0, amplitude=CALM_AMPLITUDE)
        self.assertIn(state, (VitalState.STRESS, VitalState.ESCALATE))

    def test_slower_breathing_does_not_trigger_anything(self):
        session = self.calibrated()
        self.assertEqual(session.breathe(60, rate_bpm=7.0), VitalState.NORMAL)


class TestDetectionLatency(IntegrationTestCase):
    """Latency is dominated by the FFT window, so it is a property worth pinning."""

    def test_fast_breathing_is_detected_within_the_expected_window(self):
        session = self.calibrated()
        elapsed = session.seconds_until(
            lambda s: s in (VitalState.STRESS, VitalState.ESCALATE),
            limit=60, rate_bpm=26.0)
        self.assertIsNotNone(elapsed, "rapid breathing was never detected")
        self.assertLessEqual(elapsed, 30, f"took {elapsed}s to notice fast breathing")

    def test_hard_breathing_is_detected_within_the_expected_window(self):
        session = self.calibrated()
        elapsed = session.seconds_until(
            lambda s: s == VitalState.STRESS,
            limit=90, amplitude=CALM_AMPLITUDE * 4)
        self.assertIsNotNone(elapsed, "hard breathing was never detected")
        # Slower than the rate path by design: effort must be sustained before it counts.
        self.assertLessEqual(elapsed, 45, f"took {elapsed}s to notice hard breathing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
