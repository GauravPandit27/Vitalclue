"""
VitalCue - Vision Module
Extracts the respiratory signal from a video frame.

Uses the MediaPipe Tasks PoseLandmarker to locate the chest, then measures breathing
as vertical optical flow inside that region of interest - not as the Y coordinate of
two shoulder landmarks.

Why the switch: a landmark averages two points, each quantised to about a pixel. At a
normal webcam resolution that is an excursion of roughly 0.002 in normalised coordinates
for shallow (elevated-rate) breathing - smaller than ordinary postural sway, which is
why a paced 20 brpm trial with real shoulders locked onto ~7 brpm drift instead of the
breath. Averaging flow over a few thousand chest pixels recovers that motion by roughly
the square root of the pixel count, and the validation failure that motivated this is
exactly the shallow-breathing regime.

Deliberately does not classify facial expression or emotion. Inferring emotional state
from biometric data is prohibited in workplace and education settings under EU AI Act
Article 5(1)(f), and a breathing rate is a physiological measurement rather than an
emotional inference - that distinction is what keeps this component defensible.
"""
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)

from core.model_store import MODEL_DIR, ensure_file

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

LANDMARK_COLOR = (0, 255, 0)
ROI_COLOR = (0, 200, 120)

# Fixed-size ROI crop so the optical-flow maths does not depend on how far the person
# sits from the camera. Farneback at this resolution is cheap enough to run every frame.
ROI_SIZE = (96, 72)       # width, height
# Integrate flow into a displacement. Without a leak the signal wanders with any
# residual bias in the flow estimate; with too much leak the breathing itself is
# attenuated. 0.995 keeps a ~3s memory at 20fps, long enough for a breath.
FLOW_LEAK = 0.995
# Landmark EMA used only to keep the ROI box from jittering between frames. Jitter in
# the box looks like motion inside it.
BOX_SMOOTHING = 0.7


def ensure_model(path: str = MODEL_PATH) -> str:
    """Download the pose model on first run so a fresh clone just works."""
    return ensure_file(MODEL_URL, path, "pose landmarker")


def chest_roi_from_landmarks(landmarks, frame_w: int, frame_h: int):
    """Axis-aligned box covering the upper chest, from the shoulders toward the hips.

    Returns pixel coordinates (x0, y0, x1, y1) or None if the landmarks are unusable.
    """
    left = landmarks[LEFT_SHOULDER]
    right = landmarks[RIGHT_SHOULDER]
    if left.visibility < 0.5 or right.visibility < 0.5:
        return None

    # Prefer hips when visible: they give a real lower bound for the chest rather than
    # a magic fraction of shoulder width, which fails when the person is far away.
    left_hip = landmarks[LEFT_HIP]
    right_hip = landmarks[RIGHT_HIP]
    hips_ok = left_hip.visibility > 0.5 and right_hip.visibility > 0.5

    sx0 = min(left.x, right.x)
    sx1 = max(left.x, right.x)
    sy = (left.y + right.y) / 2.0
    shoulder_span = max(sx1 - sx0, 1e-3)

    if hips_ok:
        hip_y = (left_hip.y + right_hip.y) / 2.0
        # Upper third of the torso - diaphragm motion shows up here; going all the way
        # to the hips admits arm and clothing motion that is not breathing.
        y0 = sy - 0.15 * (hip_y - sy)
        y1 = sy + 0.45 * (hip_y - sy)
    else:
        y0 = sy - 0.25 * shoulder_span
        y1 = sy + 0.70 * shoulder_span

    # Inset slightly from the shoulder tips so the box stays on the torso, not the arms.
    pad = 0.12 * shoulder_span
    x0 = sx0 + pad
    x1 = sx1 - pad

    px0 = int(np.clip(x0 * frame_w, 0, frame_w - 2))
    px1 = int(np.clip(x1 * frame_w, px0 + 2, frame_w))
    py0 = int(np.clip(y0 * frame_h, 0, frame_h - 2))
    py1 = int(np.clip(y1 * frame_h, py0 + 2, frame_h))
    if px1 - px0 < 16 or py1 - py0 < 16:
        return None
    return px0, py0, px1, py1


def mean_vertical_flow(prev_roi: np.ndarray, curr_roi: np.ndarray) -> float:
    """Mean upward-positive vertical optical flow between two ROI greyscale patches.

    Sign convention matches the old landmark signal after the waveform invert: a rising
    chest (inhale) produces a positive contribution once integrated.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_roi, curr_roi, None,
        pyr_scale=0.5, levels=2, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
    )
    # OpenCV flow: positive vy is downward in image coordinates. Inhale lifts the chest
    # toward the top of the frame, so negate.
    return float(-np.mean(flow[..., 1]))


class VisionProcessor:
    """Locates the chest with pose landmarks and reads breathing from ROI pixel motion."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._start = time.time()
        self._last_timestamp_ms = -1

        self._prev_roi = None
        self._box = None
        self._displacement = 0.0

    def _next_timestamp_ms(self) -> int:
        """VIDEO mode requires strictly increasing timestamps."""
        ts = int((time.time() - self._start) * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    def _smooth_box(self, box):
        if box is None:
            return None
        if self._box is None:
            self._box = np.array(box, dtype=float)
        else:
            self._box = BOX_SMOOTHING * self._box + (1.0 - BOX_SMOOTHING) * np.array(box)
        return tuple(int(v) for v in self._box)

    def _update_flow_signal(self, gray, box):
        """Integrate mean vertical flow inside the ROI into a displacement signal."""
        x0, y0, x1, y1 = box
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        roi = cv2.resize(crop, ROI_SIZE, interpolation=cv2.INTER_AREA)

        if self._prev_roi is None or self._prev_roi.shape != roi.shape:
            self._prev_roi = roi
            return None

        flow_y = mean_vertical_flow(self._prev_roi, roi)
        self._prev_roi = roi
        # Leaky integrator: turns per-frame velocity into a breath-shaped displacement
        # without letting residual flow bias walk the signal to infinity.
        self._displacement = FLOW_LEAK * self._displacement + flow_y
        return self._displacement

    def process_frame(self, frame_bgr):
        """
        Returns:
            annotated_frame: pose + chest ROI drawn for live lock-on feedback
            resp_signal: ROI displacement (float) or None if not yet ready
            alignment_feedback: positioning hint, or "ALIGNED"
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

        annotated = frame_bgr.copy()
        if not result.pose_landmarks:
            self._prev_roi = None
            return annotated, None, "No person detected in frame."

        lm = result.pose_landmarks[0]
        self._draw(annotated, lm)

        h, w = frame_bgr.shape[:2]
        left = lm[LEFT_SHOULDER]
        right = lm[RIGHT_SHOULDER]
        alignment_feedback = "ALIGNED"
        resp_signal = None

        if left.visibility > 0.5 and right.visibility > 0.5:
            shoulder_y = (left.y + right.y) / 2.0
            shoulder_dist = abs(left.x - right.x)

            if shoulder_y > 0.85:
                alignment_feedback = "Move back so your chest is visible."
            elif shoulder_dist > 0.7:
                alignment_feedback = "Too close. Move back slightly."
            elif shoulder_dist < 0.15:
                alignment_feedback = "Too far. Move closer to the camera."
            else:
                raw_box = chest_roi_from_landmarks(lm, w, h)
                box = self._smooth_box(raw_box)
                if box is not None:
                    x0, y0, x1, y1 = box
                    cv2.rectangle(annotated, (x0, y0), (x1, y1), ROI_COLOR, 2)
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    resp_signal = self._update_flow_signal(gray, box)
        else:
            alignment_feedback = "Ensure both shoulders are clearly visible."
            self._prev_roi = None

        return annotated, resp_signal, alignment_feedback

    @staticmethod
    def _draw(frame, landmarks):
        """Skeleton overlay. The Tasks API dropped the drawing_utils helper."""
        h, w = frame.shape[:2]
        points = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        for connection in PoseLandmarksConnections.POSE_LANDMARKS:
            start, end = connection.start, connection.end
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], LANDMARK_COLOR, 1)
        for point in points:
            cv2.circle(frame, point, 2, LANDMARK_COLOR, -1)

    def close(self):
        self.landmarker.close()
