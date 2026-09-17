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
    FATIGUE = "fatigue"        # low arousal / disengagement -> alertness path
    ESCALATE = "escalate"      # non-resolving / atypical -> hard escalation, no LLM softening


BASELINE_WINDOW_SECONDS = 15      # time to establish "this person's normal"
STRESS_Z_THRESHOLD = 1.75         # deviation from personal baseline to flag stress
RECOVERY_SECONDS = 45             # time allowed for intervention to bring rate back down
ESCALATION_Z_THRESHOLD = 3.0      # deviation severe/persistent enough to skip calming


class BaselineEngine:
    """
    Tracks a rolling personal baseline for respiration rate and drives a simple,
    explainable state machine. Deliberately rule-based (not ML) so the decision
    logic is a one-sentence answer under questioning, not a black box: escalation
    is decided here, before the LLM is ever invoked - the LLM only phrases the
    calming response once this gate has already ruled escalation out.
    """

    def __init__(self):
        self.baseline_samples = deque()
        self.baseline_start = None
        self.state = VitalState.CALIBRATING
        self.state_entered_at = time.time()
        self.baseline_mean = None
        self.baseline_std = None

    def _set_state(self, new_state: VitalState):
        if new_state != self.state:
            self.state = new_state
            self.state_entered_at = time.time()

    def update(self, bpm, confidence: float):
        """
        Feed a new respiratory-rate reading. Returns the current VitalState.
        Low-confidence readings are ignored rather than treated as data - an
        unreliable frame should never move the baseline or trigger a state change.
        """
        if bpm is None or confidence < 0.05:
            return self.state

        now = time.time()

        if self.state == VitalState.CALIBRATING:
            if not self.baseline_samples:
                self.baseline_start = now
            self.baseline_samples.append(bpm)
            if now - self.baseline_start >= BASELINE_WINDOW_SECONDS and len(self.baseline_samples) >= 5:
                self._finalize_baseline()
                self._set_state(VitalState.NORMAL)
            return self.state

        z = self._z_score(bpm)

        if self.state == VitalState.NORMAL:
            if z >= ESCALATION_Z_THRESHOLD:
                self._set_state(VitalState.ESCALATE)
            elif z >= STRESS_Z_THRESHOLD:
                self._set_state(VitalState.STRESS)

        elif self.state == VitalState.STRESS:
            elapsed = now - self.state_entered_at
            if z < STRESS_Z_THRESHOLD * 0.6:
                self._set_state(VitalState.NORMAL)
            elif z >= ESCALATION_Z_THRESHOLD or elapsed >= RECOVERY_SECONDS:
                # Either it got worse, or the calming path had its window and didn't
                # work - hard gate, escalation is never decided by the LLM.
                self._set_state(VitalState.ESCALATE)

        elif self.state == VitalState.ESCALATE:
            # Requires a return below stress threshold to leave escalation
            if z < STRESS_Z_THRESHOLD:
                self._set_state(VitalState.NORMAL)

        return self.state

    def _finalize_baseline(self):
        values = list(self.baseline_samples)
        self.baseline_mean = statistics.mean(values)
        self.baseline_std = statistics.pstdev(values) or 1.0  # avoid div-by-zero

    def _z_score(self, bpm: float) -> float:
        if self.baseline_mean is None or self.baseline_std is None:
            return 0.0
        return abs(bpm - self.baseline_mean) / self.baseline_std

    def calibration_progress(self) -> float:
        """0.0 - 1.0 progress through the calibration window, for the UI countdown."""
        if not self.baseline_start:
            return 0.0
        elapsed = time.time() - self.baseline_start
        return min(1.0, elapsed / BASELINE_WINDOW_SECONDS)
