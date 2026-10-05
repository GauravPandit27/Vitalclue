"""
VitalCue - Baseline & State Engine
Personal-baseline z-score detection + state machine driving the intervention flow.
"""
import statistics
import time
from collections import deque
from enum import Enum


class VitalState(Enum):
    CALIBRATING = "calibrating"
    NORMAL = "normal"
    STRESS = "stress"          # acute spike, stress-pattern -> calming path
    FATIGUE = "fatigue"        # low arousal / disengagement -> not yet implemented
    ESCALATE = "escalate"      # non-resolving / atypical -> hard escalation, no LLM softening


# Calibration must outlast one full FFT window, otherwise every baseline sample is
# computed from largely the same underlying data and the resulting standard deviation
# measures FFT jitter rather than the person's real breathing variability.
BASELINE_WINDOW_SECONDS = 60
BASELINE_SAMPLE_INTERVAL = 5.0    # subsample so baseline points are less autocorrelated
MIN_BASELINE_SAMPLES = 8

# Demo/short-session profile. Still longer than one FFT window, just barely.
DEMO_BASELINE_WINDOW_SECONDS = 25
DEMO_BASELINE_SAMPLE_INTERVAL = 2.5
DEMO_MIN_BASELINE_SAMPLES = 6

# Floor on the baseline spread. Overlapping FFT windows make consecutive readings
# highly correlated, so pstdev can collapse toward zero; without this floor an
# ordinary 2 BPM shift divides into a z-score above the escalation threshold and
# the calming path gets skipped entirely.
MIN_BASELINE_STD = 1.5

# Effort baseline. The absolute excursion depends on build, posture and distance from
# the camera, so like the rate it is only ever compared against the same person's own
# calibration. The spread floor is a fraction rather than an absolute because the
# quantity is scale-dependent: 20% of the baseline mean is small enough to stay
# sensitive, large enough that a near-motionless calibration cannot make every breath
# afterwards look like a huge deviation.
MIN_EFFORT_STD_FRACTION = 0.20

# Breathing harder has to be *sustained* before it means anything. Reaching for a coffee
# or shifting in the seat moves the shoulders far more than any breath, and would trip an
# instantaneous threshold every time.
EFFORT_Z_THRESHOLD = 2.0
EFFORT_EXIT_Z = 1.2
EFFORT_SUSTAIN_SECONDS = 10.0

# Large excursions only mean distress if the person is not also breathing slowly. Deep
# *and slow* is the calming pattern the intervention itself coaches, and it produces a
# very high effort reading: without this guard, following the guidance drives effort up,
# re-triggers STRESS, and the app fights its own intervention. Measured, paced breathing
# at 7 brpm against a 14 brpm baseline reached effort z +8.7 while rate z sat at -4.7.
EFFORT_REQUIRES_RATE_Z_ABOVE = -0.5

STRESS_Z_THRESHOLD = 1.75         # deviation above personal baseline to flag stress
STRESS_EXIT_Z = 1.05              # lower exit threshold -> hysteresis, no flip-flopping
ESCALATION_Z_THRESHOLD = 3.0      # deviation severe enough to skip calming
ESCALATE_EXIT_Z = 1.40            # must come well down before leaving escalation

RECOVERY_SECONDS = 45             # time allowed for the intervention to show improvement
MAX_STRESS_SECONDS = 120          # absolute cap regardless of improvement
IMPROVEMENT_MARGIN = 0.35         # z-score drop that counts as "the intervention is working"
MIN_STATE_SECONDS = 5.0           # dwell time before any state may be left


class BaselineEngine:
    """
    Tracks a personal baseline for respiration rate and drives a simple, explainable
    state machine. Deliberately rule-based (not ML) so the decision logic is a
    one-sentence answer under questioning, not a black box: escalation is decided
    here, before the LLM is ever invoked - the LLM only phrases the calming response
    once this gate has already ruled escalation out.

    Deviation is directional. Only rates *above* baseline drive STRESS/ESCALATE;
    breathing more slowly than usual is not a stress signal.

    Two signals, either of which can trigger. Rate catches rapid shallow breathing,
    the classic anxiety pattern. Effort catches breathing hard without breathing fast,
    which is completely invisible to a rate estimator - the same shoulders moving five
    times as far at the same speed produce an identical BPM. They are kept separate
    rather than merged into one score so that when the system fires, which signal fired
    it is still answerable.
    """

    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        if demo_mode:
            self.baseline_window = DEMO_BASELINE_WINDOW_SECONDS
            self.sample_interval = DEMO_BASELINE_SAMPLE_INTERVAL
            self.min_samples = DEMO_MIN_BASELINE_SAMPLES
        else:
            self.baseline_window = BASELINE_WINDOW_SECONDS
            self.sample_interval = BASELINE_SAMPLE_INTERVAL
            self.min_samples = MIN_BASELINE_SAMPLES

        self.baseline_samples = deque()
        self.baseline_effort_samples = deque()
        self.baseline_start = None
        self.last_baseline_sample_at = 0.0

        self.state = VitalState.CALIBRATING
        self.state_entered_at = time.time()
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_effort_mean = None
        self.baseline_effort_std = None

        # Set on every update so the UI can show what the decision is actually based on.
        self.last_z = 0.0
        self.last_effort_z = None
        self.effort_high_since = None

        # Intervention outcome tracking. Whether the guided breathing actually
        # returned the person to their baseline is the metric this project is
        # really about, so it is recorded rather than inferred after the fact.
        self.stress_entered_at = None
        self.stress_peak_z = None
        self.recovery_events = []

    def _set_state(self, new_state: VitalState):
        if new_state != self.state:
            self.state = new_state
            self.state_entered_at = time.time()

    def update(self, bpm, confidence: float, effort=None):
        """
        Feed a new respiratory-rate reading, optionally with a breathing-effort
        measure. Returns the current VitalState.

        Low-confidence readings are ignored rather than treated as data - an
        unreliable frame should never move the baseline or trigger a state change.
        """
        if bpm is None or confidence < 0.05:
            return self.state

        now = time.time()

        if self.state == VitalState.CALIBRATING:
            self._collect_baseline(bpm, effort, now)
            return self.state

        z = self._z_score(bpm)
        effort_z = self._effort_z(effort)
        self.last_z = z
        self.last_effort_z = effort_z

        effort_sustained = self._effort_sustained(effort_z, z, now)
        # One severity number for peak and improvement tracking, so an episode opened by
        # effort is judged on effort rather than on a rate that never moved.
        severity = max(z, effort_z if effort_z is not None else z)
        dwell = now - self.state_entered_at

        if self.state == VitalState.NORMAL:
            if z >= ESCALATION_Z_THRESHOLD:
                self._enter_stress(severity, now)
                self._set_state(VitalState.ESCALATE)
            elif z >= STRESS_Z_THRESHOLD or effort_sustained:
                self._enter_stress(severity, now)
                self._set_state(VitalState.STRESS)

        elif self.state == VitalState.STRESS:
            # Compare against the episode peak *before* folding this reading in,
            # otherwise the peak tracks the current value and improvement can
            # never register.
            improving = self._is_improving(severity)
            self.stress_peak_z = max(self.stress_peak_z, severity)
            elapsed = now - self.stress_entered_at
            calm = (z < STRESS_EXIT_Z
                    and not self._effort_concerning(effort_z, z, EFFORT_EXIT_Z))

            if calm and dwell >= MIN_STATE_SECONDS:
                self._record_recovery(now, resolved=True)
                self._set_state(VitalState.NORMAL)
            elif z >= ESCALATION_Z_THRESHOLD:
                self._set_state(VitalState.ESCALATE)
            elif elapsed >= MAX_STRESS_SECONDS:
                self._record_recovery(now, resolved=False)
                self._set_state(VitalState.ESCALATE)
            elif elapsed >= RECOVERY_SECONDS and not improving:
                # The calming path had its window and the rate is not coming down.
                # Hard gate - escalation is never decided by the LLM.
                self._record_recovery(now, resolved=False)
                self._set_state(VitalState.ESCALATE)

        elif self.state == VitalState.ESCALATE:
            calm = (z < ESCALATE_EXIT_Z
                    and not self._effort_concerning(effort_z, z, EFFORT_EXIT_Z))
            if calm and dwell >= MIN_STATE_SECONDS:
                # No-op unless escalation was reached straight from NORMAL; the
                # STRESS path already closed its episode on the way through.
                self._record_recovery(now, resolved=True)
                self._set_state(VitalState.NORMAL)

        return self.state

    def _collect_baseline(self, bpm: float, effort, now: float):
        if not self.baseline_samples:
            self.baseline_start = now
            self.baseline_samples.append(bpm)
            if effort is not None:
                self.baseline_effort_samples.append(effort)
            self.last_baseline_sample_at = now
            return

        if now - self.last_baseline_sample_at >= self.sample_interval:
            self.baseline_samples.append(bpm)
            if effort is not None:
                self.baseline_effort_samples.append(effort)
            self.last_baseline_sample_at = now

        window_elapsed = now - self.baseline_start >= self.baseline_window
        if window_elapsed and len(self.baseline_samples) >= self.min_samples:
            self._finalize_baseline()
            self._set_state(VitalState.NORMAL)

    def _finalize_baseline(self):
        values = list(self.baseline_samples)
        self.baseline_mean = statistics.mean(values)
        self.baseline_std = max(statistics.pstdev(values), MIN_BASELINE_STD)

        efforts = list(self.baseline_effort_samples)
        if len(efforts) >= 2:
            self.baseline_effort_mean = statistics.mean(efforts)
            self.baseline_effort_std = max(
                statistics.pstdev(efforts),
                MIN_EFFORT_STD_FRACTION * self.baseline_effort_mean,
            )

    def _effort_z(self, effort):
        """Signed deviation of breathing excursion above baseline, or None."""
        if effort is None or not self.baseline_effort_std:
            return None
        return (effort - self.baseline_effort_mean) / self.baseline_effort_std

    @staticmethod
    def _effort_concerning(effort_z, rate_z: float, threshold: float) -> bool:
        """Is this excursion a distress signal, or just someone breathing deeply?

        The rate guard applies on the way out as well as the way in. Without it, a
        person who follows the calming guidance raises their effort reading and can
        never satisfy the exit condition - they would be held in STRESS precisely
        because the intervention was working.
        """
        if effort_z is None or rate_z <= EFFORT_REQUIRES_RATE_Z_ABOVE:
            return False
        return effort_z >= threshold

    def _effort_sustained(self, effort_z, rate_z: float, now: float) -> bool:
        """True once effort has stayed above threshold long enough to mean something."""
        if not self._effort_concerning(effort_z, rate_z, EFFORT_Z_THRESHOLD):
            self.effort_high_since = None
            return False
        if self.effort_high_since is None:
            self.effort_high_since = now
        return now - self.effort_high_since >= EFFORT_SUSTAIN_SECONDS

    def effort_ratio(self, effort):
        """Effort as a multiple of this person's calibrated normal, for display."""
        if effort is None or not self.baseline_effort_mean:
            return None
        return effort / self.baseline_effort_mean

    def _enter_stress(self, z: float, now: float):
        self.stress_entered_at = now
        self.stress_peak_z = z

    def _is_improving(self, z: float) -> bool:
        """True when the rate has come down meaningfully from this episode's peak."""
        if self.stress_peak_z is None:
            return False
        return z <= self.stress_peak_z - IMPROVEMENT_MARGIN

    def _record_recovery(self, now: float, resolved: bool):
        if self.stress_entered_at is None:
            return
        self.recovery_events.append(
            {
                "entered_at": self.stress_entered_at,
                "duration_seconds": now - self.stress_entered_at,
                "resolved": resolved,
            }
        )
        self.stress_entered_at = None
        self.stress_peak_z = None

    def _z_score(self, bpm: float) -> float:
        """Signed deviation above baseline. Negative values mean slower than usual."""
        if self.baseline_mean is None or self.baseline_std is None:
            return 0.0
        return (bpm - self.baseline_mean) / self.baseline_std

    def calibration_progress(self) -> float:
        """0.0 - 1.0 progress through the calibration window, for the UI countdown."""
        if not self.baseline_start:
            return 0.0
        elapsed = time.time() - self.baseline_start
        return min(1.0, elapsed / self.baseline_window)

    def recovery_summary(self):
        """Aggregate intervention outcome for the session, or None if no episodes yet."""
        if not self.recovery_events:
            return None
        resolved = [e for e in self.recovery_events if e["resolved"]]
        return {
            "episodes": len(self.recovery_events),
            "resolved": len(resolved),
            "mean_recovery_seconds": (
                statistics.mean(e["duration_seconds"] for e in resolved) if resolved else None
            ),
        }
