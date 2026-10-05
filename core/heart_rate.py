"""
VitalCue - Heart Rate via rPPG

Estimates pulse rate from facial video using PhysNet, a pretrained spatio-temporal CNN
from rPPG-Toolbox. The camera picks up sub-perceptual colour changes in facial skin as
blood volume varies with each beat; the network maps a stack of frames to that pulse
waveform, and the rate is the dominant frequency of the waveform.

This is a wellness signal, not a medical measurement. Camera-based pulse rate is exactly
what NuraLogix's Anura SDK holds FDA 510(k) clearance for (K253650, June 2026), and that
clearance requires the subject relaxed, still and seated upright. Nothing here is
cleared, validated, or suitable for a clinical claim.

Weights: UBFC-rPPG_PhysNet_DiffNormalized.pth, trained on UBFC-rPPG (webcam recordings
of seated subjects) - the closest public training distribution to this use case.
"""
import os
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode

from core.model_store import MODEL_DIR, ensure_file

PHYSNET_PATH = os.path.join(MODEL_DIR, "physnet_ubfc.pth")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")

PHYSNET_URL = (
    "https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/"
    "UBFC-rPPG_PhysNet_DiffNormalized.pth"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

# PhysNet's temporal pooling is fixed at construction, so the chunk length is not free.
# It can be rebuilt at other lengths, but cost scales with T and T=128 already measures
# ~2s per forward pass on a laptop CPU.
CHUNK_FRAMES = 128
CROP_SIZE = 128

# The network learned temporal dynamics at 30fps, and 128 frames at 30fps is the exact
# configuration it was trained and benchmarked at. A laptop webcam running MediaPipe
# manages nearer 11fps, so frames are buffered by *time* and resampled onto a 30fps grid
# before inference. Feeding native-rate frames instead would present the heartbeat to the
# network almost three times slower than anything it saw in training.
TARGET_FPS = 30.0
WINDOW_SECONDS = CHUNK_FRAMES / TARGET_FPS   # 4.27s
MIN_FRAMES_IN_WINDOW = 40                    # refuse to upsample from too little real data

# Plausible human pulse range. Narrower than the respiratory band, and deliberately
# excluding the extremes: at these window lengths a peak out at 200 bpm is far more
# likely to be motion than a real heartbeat.
HR_LOW_HZ = 0.7    # 42 bpm
HR_HIGH_HZ = 3.0   # 180 bpm

# A forward pass measures ~2s on a laptop CPU. It runs on a worker thread so it never
# stalls the frame loop, but the interval still has to exceed it: any shorter and the
# worker runs continuously, competing with MediaPipe for CPU on the capture thread.
INFER_INTERVAL_SECONDS = 5.0

# PhysNet would otherwise grab every core and leave the pose landmarker fighting for CPU
# on the frame loop. Measured on a 14-core machine, 2 threads is as fast as 6 for this
# model, so the extra cores buy nothing here and are better left to the camera pipeline.
TORCH_THREADS = 2

FACE_DETECT_EVERY = 10         # redetect periodically, smooth the box in between
BOX_SMOOTHING = 0.7            # EMA on the crop box; jitter in the ROI looks like signal
# A 4.27s window resolves only ~14 bpm per FFT bin. Parabolic interpolation recovers
# most of that, and a median across chunks absorbs the rest - a pulse rate does not
# change fast enough for the smoothing to hide anything real.
HR_HISTORY = 8


class HeartRateProcessor:
    """Buffers face crops and estimates pulse rate with PhysNet."""

    def __init__(self, chunk_frames: int = CHUNK_FRAMES, infer_interval: float = INFER_INTERVAL_SECONDS):
        self.chunk_frames = chunk_frames
        self.infer_interval = infer_interval

        self.crops = deque()
        self.timestamps = deque()
        self.hr_history = deque(maxlen=HR_HISTORY)

        self._box = None
        self._frame_count = 0
        self._last_inference = 0.0
        self._cached = (None, 0.0)
        self._bvp = None
        self._start = time.time()
        self._last_ts_ms = -1

        # Inference runs on a worker so the caller's frame loop never blocks on it: the
        # camera keeps reading and the UI keeps redrawing while the pulse estimate
        # refreshes behind it. Only one worker runs at a time, so the model is never
        # entered concurrently.
        self._lock = threading.Lock()
        self._worker = None
        self._closed = False

        ensure_file(FACE_MODEL_URL, FACE_MODEL_PATH, "face detector")
        self.detector = FaceDetector.create_from_options(
            FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=RunningMode.VIDEO,
                min_detection_confidence=0.5,
            )
        )
        self._model = None  # torch import is slow; defer until first use

    def _load_model(self):
        if self._model is not None:
            return self._model
        import torch  # noqa: PLC0415 - deferred so the app starts without paying for it

        from core.models.physnet import PhysNet_padding_Encoder_Decoder_MAX

        torch.set_num_threads(TORCH_THREADS)
        ensure_file(PHYSNET_URL, PHYSNET_PATH, "PhysNet weights")
        model = PhysNet_padding_Encoder_Decoder_MAX(frames=self.chunk_frames)
        model.load_state_dict(torch.load(PHYSNET_PATH, map_location="cpu", weights_only=True))
        model.eval()
        self._torch = torch
        self._model = model
        return model

    def _next_timestamp_ms(self) -> int:
        ts = int((time.time() - self._start) * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def _detect_box(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.detector.detect_for_video(image, self._next_timestamp_ms())
        if not result.detections:
            return None
        bb = result.detections[0].bounding_box
        return np.array([bb.origin_x, bb.origin_y, bb.width, bb.height], dtype=float)

    def add_frame(self, frame_bgr):
        """Detect (or reuse) the face box, crop, and buffer. Returns the crop box or None."""
        if self._frame_count % FACE_DETECT_EVERY == 0 or self._box is None:
            box = self._detect_box(frame_bgr)
            if box is not None:
                self._box = box if self._box is None else (
                    BOX_SMOOTHING * self._box + (1.0 - BOX_SMOOTHING) * box
                )
        self._frame_count += 1

        if self._box is None:
            return None

        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = self._box
        x0, y0 = int(max(0, x)), int(max(0, y))
        x1, y1 = int(min(w, x + bw)), int(min(h, y + bh))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None

        crop = cv2.resize(frame_bgr[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE),
                          interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Held while mutating, because the inference worker snapshots these buffers.
        with self._lock:
            self.crops.append(rgb)
            self.timestamps.append(time.time())
            while (len(self.timestamps) > 1
                   and self.timestamps[-1] - self.timestamps[0] > WINDOW_SECONDS):
                self.timestamps.popleft()
                self.crops.popleft()
        return (x0, y0, x1, y1)

    def _select_frames(self):
        """Nearest-neighbour resample of the buffered crops onto a 30fps grid.

        Nearest rather than blended: interpolating between frames would average away
        part of the very colour change the pulse lives in.

        Returns references, not a stacked array. Stacking 128 crops copies ~25MB, and
        doing that under the lock on the caller's thread showed up as a visible hitch;
        the worker stacks them instead. Safe because crops are never mutated in place -
        add_frame always appends a fresh array.
        """
        real = np.array(self.timestamps)
        target = np.linspace(real[0], real[-1], self.chunk_frames)
        idx = np.abs(real[None, :] - target[:, None]).argmin(axis=1)
        return [self.crops[i] for i in idx]

    def _resampled_chunk(self):
        return np.stack(self._select_frames())

    @staticmethod
    def _diff_normalize(frames: np.ndarray) -> np.ndarray:
        """Frame-to-frame normalised difference, as PhysNet was trained on.

        Dividing the difference by the sum cancels illumination scale, leaving the
        relative colour change that carries the pulse.
        """
        diff = (frames[1:] - frames[:-1]) / (frames[1:] + frames[:-1] + 1e-7)
        std = np.std(diff)
        if std > 0:
            diff = diff / std
        diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        # Pad back to the original length so the chunk matches the model's fixed T.
        return np.concatenate([diff, np.zeros_like(frames[:1])], axis=0)

    @staticmethod
    def _bpm_from_bvp(bvp: np.ndarray, fs: float):
        n = len(bvp)
        signal = bvp - np.mean(bvp)
        windowed = signal * np.hanning(n)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        power = np.abs(np.fft.rfft(windowed)) ** 2

        band = np.flatnonzero((freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ))
        if band.size == 0:
            return None, 0.0

        peak = int(band[np.argmax(power[band])])
        # Same sub-bin refinement as the respiratory path: a 4.27s chunk resolves only
        # about 14 bpm per bin, which is far too coarse to report a pulse rate from.
        if 0 < peak < len(power) - 1:
            a, b, c = np.log(power[peak - 1: peak + 2] + 1e-20)
            denom = a - 2.0 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
            delta = delta if abs(delta) <= 0.5 else 0.0
        else:
            delta = 0.0
        peak_freq = freqs[peak] + delta * (freqs[1] - freqs[0])

        confidence = float(power[peak] / (np.sum(power[band]) + 1e-9))
        return float(peak_freq * 60.0), confidence

    def estimate_rate(self):
        """Returns the latest (bpm, confidence) immediately, never blocking.

        Inference happens on a worker thread, so this returns the previous estimate
        while a new one is being computed. That is not a compromise for a pulse rate:
        the value is already a median over several seconds of video.
        """
        with self._lock:
            cached = self._cached
            busy = self._worker is not None and self._worker.is_alive()
            due = time.time() - self._last_inference >= self.infer_interval
            if self._closed or busy or not due or not self._is_ready_locked():
                return cached

            self._last_inference = time.time()
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration <= 0:
                return cached
            selected = self._select_frames()  # references only; cheap under the lock
            self._worker = threading.Thread(
                target=self._infer, args=(selected, duration), daemon=True)
            self._worker.start()
        return cached

    def _infer(self, selected, duration):
        """Runs on the worker thread. Never raises into the caller's frame loop."""
        try:
            # After resampling the chunk spans the same wall-clock time with a fixed
            # frame count, so this lands on ~30 regardless of the camera's real rate.
            fs = self.chunk_frames / duration

            normalized = self._diff_normalize(np.stack(selected))
            tensor = np.transpose(normalized, (3, 0, 1, 2))  # (C, T, H, W)

            model = self._load_model()
            torch = self._torch
            with torch.no_grad():
                bvp, _, _, _ = model(torch.from_numpy(tensor[None]).float())
            bvp = bvp.squeeze().numpy()

            bpm, confidence = self._bpm_from_bvp(bvp, fs)
            if bpm is None:
                return

            with self._lock:
                self.hr_history.append(bpm)
                # Median over recent chunks: a single 4.27s window is easily thrown off
                # by a head movement, and a pulse rate does not change that fast.
                self._cached = (float(np.median(self.hr_history)), confidence)
                self._bvp = bvp
        except Exception as exc:  # noqa: BLE001 - a dropped estimate must not kill the UI
            print(f"Heart rate inference failed: {type(exc).__name__}: {exc}")

    def get_waveform(self):
        """Most recent pulse waveform, normalised for display."""
        with self._lock:
            bvp = self._bvp
        if bvp is None:
            return []
        centered = bvp - np.mean(bvp)
        peak = np.max(np.abs(centered))
        return (centered / peak).tolist() if peak > 0 else []

    def _is_ready_locked(self) -> bool:
        if len(self.crops) < MIN_FRAMES_IN_WINDOW:
            return False
        return self.timestamps[-1] - self.timestamps[0] >= WINDOW_SECONDS * 0.95

    def is_ready(self) -> bool:
        with self._lock:
            return self._is_ready_locked()

    def progress(self) -> float:
        with self._lock:
            if len(self.timestamps) < 2:
                return 0.0
            span = (self.timestamps[-1] - self.timestamps[0]) / WINDOW_SECONDS
            frames = len(self.crops) / MIN_FRAMES_IN_WINDOW
        return min(1.0, min(span, frames))

    def close(self):
        with self._lock:
            self._closed = True
            worker = self._worker
        if worker is not None:
            worker.join(timeout=5.0)
        self.detector.close()
