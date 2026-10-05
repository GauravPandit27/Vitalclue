"""
State machine tests with a simulated clock.

The case that matters most is the first one: before the standard-deviation floor
existed, a realistic calibration produced a spread near zero, so an ordinary rise
in breathing rate scored past the escalation threshold and jumped straight to
ESCALATE — skipping the calming intervention the whole product is built around.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import engine as engine_module  # noqa: E402
from core.engine import BaselineEngine, VitalState  # noqa: E402


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self._real_time = engine_module.time.time
        engine_module.time.time = self.clock

    def tearDown(self):
        engine_module.time.time = self._real_time

    def calibrate(self, eng, rate=14.0, jitter=0.2, effort=None):
        """Feed calibration readings that jitter only slightly, as overlapping
        FFT windows really do, then confirm the baseline was accepted."""
        for i in range(60):
            sign = 1 if i % 2 else -1
            eng.update(rate + sign * jitter, confidence=0.5,
                       effort=None if effort is None else effort * (1 + sign * 0.02))
            self.clock.advance(2.0)
        self.assertIsNotNone(eng.baseline_mean, "calibration did not complete")
        return eng

    def hold_effort(self, eng, effort, seconds, rate=None, step=1.0):
        """Feed a steady elevated effort for a while at an unchanged breathing rate."""
        rate = eng.baseline_mean if rate is None else rate
        state = eng.state
        for _ in range(int(seconds / step)):
            state = eng.update(rate, confidence=0.5, effort=effort)
            self.clock.advance(step)
        return state

    @staticmethod
    def at_z(eng, z):
        """The bpm reading that lands exactly at the given z-score."""
        return eng.baseline_mean + z * eng.baseline_std


class TestBaselineFloor(EngineTestCase):
    def test_tight_calibration_does_not_inflate_z_scores(self):
        eng = self.calibrate(BaselineEngine())
        self.assertGreaterEqual(eng.baseline_std, engine_module.MIN_BASELINE_STD)

    def test_moderate_rise_triggers_stress_not_escalation(self):
        eng = self.calibrate(BaselineEngine())
        state = eng.update(18.0, confidence=0.5)
        self.assertEqual(state, VitalState.STRESS)

    def test_severe_rise_still_escalates_directly(self):
        eng = self.calibrate(BaselineEngine())
        state = eng.update(30.0, confidence=0.5)
        self.assertEqual(state, VitalState.ESCALATE)


class TestEffortTrigger(EngineTestCase):
    """Breathing hard without breathing fast is invisible to the rate estimator, so
    these cases all hold the rate exactly at baseline and vary only the excursion."""

    BASE_EFFORT = 0.01

    def calibrated(self):
        return self.calibrate(BaselineEngine(), effort=self.BASE_EFFORT)

    def test_sustained_hard_breathing_triggers_stress_at_an_unchanged_rate(self):
        eng = self.calibrated()
        state = self.hold_effort(eng, self.BASE_EFFORT * 3, seconds=20)
        self.assertEqual(state, VitalState.STRESS)

    def test_a_brief_spike_does_not_trigger(self):
        """Reaching for a coffee moves the shoulders more than any breath does."""
        eng = self.calibrated()
        state = self.hold_effort(eng, self.BASE_EFFORT * 5, seconds=4)
        self.assertEqual(state, VitalState.NORMAL)

    def test_the_sustain_timer_resets_when_effort_drops_back(self):
        eng = self.calibrated()
        self.hold_effort(eng, self.BASE_EFFORT * 3, seconds=6)
        self.hold_effort(eng, self.BASE_EFFORT, seconds=3)
        state = self.hold_effort(eng, self.BASE_EFFORT * 3, seconds=6)
        self.assertEqual(state, VitalState.NORMAL,
                         "an interrupted spell of hard breathing should start over")

    def test_breathing_more_gently_than_baseline_is_not_stress(self):
        eng = self.calibrated()
        state = self.hold_effort(eng, self.BASE_EFFORT * 0.3, seconds=30)
        self.assertEqual(state, VitalState.NORMAL)

    def test_effort_episode_resolves_when_breathing_softens(self):
        eng = self.calibrated()
        self.hold_effort(eng, self.BASE_EFFORT * 3, seconds=20)
        state = self.hold_effort(eng, self.BASE_EFFORT, seconds=10)
        self.assertEqual(state, VitalState.NORMAL)
        self.assertTrue(eng.recovery_events[-1]["resolved"])

    def test_rate_still_triggers_on_its_own_without_any_effort_data(self):
        eng = self.calibrate(BaselineEngine())
        self.assertIsNone(eng.baseline_effort_mean)
        self.assertEqual(eng.update(18.0, confidence=0.5), VitalState.STRESS)

    def test_effort_baseline_spread_has_a_proportional_floor(self):
        """A near-motionless calibration must not make every later breath a huge z."""
        eng = self.calibrated()
        self.assertGreaterEqual(
            eng.baseline_effort_std,
            engine_module.MIN_EFFORT_STD_FRACTION * eng.baseline_effort_mean * 0.999,
        )

    def test_effort_z_is_reported_for_the_ui(self):
        eng = self.calibrated()
        eng.update(eng.baseline_mean, confidence=0.5, effort=self.BASE_EFFORT * 3)
        self.assertGreater(eng.last_effort_z, 2.0)
        self.assertAlmostEqual(eng.effort_ratio(self.BASE_EFFORT * 3), 3.0, delta=0.1)


class TestDirectionality(EngineTestCase):
    def test_breathing_slower_than_baseline_is_not_stress(self):
        eng = self.calibrate(BaselineEngine())
        state = eng.update(8.0, confidence=0.5)
        self.assertEqual(state, VitalState.NORMAL)


class TestHysteresis(EngineTestCase):
    def test_state_holds_between_entry_and_exit_thresholds(self):
        eng = self.calibrate(BaselineEngine())
        eng.update(18.0, confidence=0.5)
        self.clock.advance(10.0)
        # z sits below the stress entry threshold but above the exit threshold.
        midband = eng.baseline_mean + 1.4 * eng.baseline_std
        self.assertEqual(eng.update(midband, confidence=0.5), VitalState.STRESS)

    def test_recovery_below_exit_threshold_returns_to_normal(self):
        eng = self.calibrate(BaselineEngine())
        eng.update(18.0, confidence=0.5)
        self.clock.advance(10.0)
        self.assertEqual(eng.update(14.0, confidence=0.5), VitalState.NORMAL)


class TestEscalationTiming(EngineTestCase):
    def test_sustained_unimproving_stress_escalates(self):
        eng = self.calibrate(BaselineEngine())
        eng.update(18.0, confidence=0.5)
        self.clock.advance(engine_module.RECOVERY_SECONDS + 1)
        self.assertEqual(eng.update(18.0, confidence=0.5), VitalState.ESCALATE)

    def test_improving_stress_is_given_more_time(self):
        eng = self.calibrate(BaselineEngine())
        self.assertEqual(eng.update(self.at_z(eng, 2.5), confidence=0.5), VitalState.STRESS)
        self.clock.advance(engine_module.RECOVERY_SECONDS + 1)
        # Coming down, but not yet under the exit threshold — keep coaching.
        self.assertEqual(eng.update(self.at_z(eng, 1.5), confidence=0.5), VitalState.STRESS)

    def test_absolute_cap_escalates_even_while_improving(self):
        eng = self.calibrate(BaselineEngine())
        eng.update(self.at_z(eng, 2.5), confidence=0.5)
        for _ in range(6):
            self.clock.advance(25.0)
            state = eng.update(self.at_z(eng, 1.5), confidence=0.5)
        self.assertEqual(state, VitalState.ESCALATE)


class TestRecoveryTracking(EngineTestCase):
    def test_resolved_episode_is_recorded_with_duration(self):
        eng = self.calibrate(BaselineEngine())
        eng.update(18.0, confidence=0.5)
        self.clock.advance(20.0)
        eng.update(14.0, confidence=0.5)

        self.assertEqual(len(eng.recovery_events), 1)
        event = eng.recovery_events[0]
        self.assertTrue(event["resolved"])
        self.assertAlmostEqual(event["duration_seconds"], 20.0, places=1)

        summary = eng.recovery_summary()
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["resolved"], 1)


class TestQualityGating(EngineTestCase):
    def test_low_confidence_readings_are_ignored(self):
        eng = self.calibrate(BaselineEngine())
        self.assertEqual(eng.update(40.0, confidence=0.01), VitalState.NORMAL)

    def test_demo_mode_calibrates_faster(self):
        eng = BaselineEngine(demo_mode=True)
        for _ in range(30):
            eng.update(14.0, confidence=0.5)
            self.clock.advance(2.0)
        self.assertIsNotNone(eng.baseline_mean)
        self.assertLess(eng.baseline_window, engine_module.BASELINE_WINDOW_SECONDS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
